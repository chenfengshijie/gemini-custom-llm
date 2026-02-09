/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
} from '@google/genai';
import OpenAI from 'openai';
import type { ContentGenerator } from '../core/contentGenerator.js';
import { extractToolFunctions, normalizeContents } from './util.js';
import type { CustomLLMContentGeneratorConfig, ToolCallMap } from './types.js';
import { ModelConverter } from './converter.js';
import { debugLogger } from '../utils/debugLogger.js';

interface CustomLLMOptions {
  userAgent?: string;
}

export class CustomLLMContentGenerator implements ContentGenerator {
  private readonly client: OpenAI;
  private readonly apiKey: string;
  private readonly baseURL: string;
  private readonly modelName: string;
  private readonly temperature: number;
  private readonly maxTokens: number;
  private readonly topP: number;
  private readonly config: CustomLLMContentGeneratorConfig;

  constructor(options: CustomLLMOptions = {}) {
    this.apiKey = process.env['CUSTOM_LLM_API_KEY'] ?? '';
    this.baseURL = process.env['CUSTOM_LLM_ENDPOINT'] ?? '';
    this.modelName = (process.env['CUSTOM_LLM_MODEL_NAME'] ?? '').trim();
    this.temperature = Number(process.env['CUSTOM_LLM_TEMPERATURE'] ?? 0);
    this.maxTokens = Number(process.env['CUSTOM_LLM_MAX_TOKENS'] ?? 8192);
    this.topP = Number(process.env['CUSTOM_LLM_TOP_P'] ?? 1);

    this.config = {
      model: this.modelName,
      temperature: this.temperature,
      max_tokens: this.maxTokens,
      top_p: this.topP,
      stream_options: {
        include_usage: true,
      },
    };

    this.client = new OpenAI({
      apiKey: this.apiKey,
      baseURL: this.baseURL || undefined,
      defaultHeaders: options.userAgent
        ? { 'User-Agent': options.userAgent }
        : undefined,
    });
  }

  async generateContent(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<GenerateContentResponse> {
    const messages = ModelConverter.toOpenAIMessages(request);
    const tools = extractToolFunctions(request.config);
    const resolvedModel = this.getResolvedModel(request.model);
    if (shouldDebugApi(userPromptId)) {
      debugLogger.log(
        '[custom-llm-api-request]',
        JSON.stringify(
          {
            promptId: userPromptId,
            baseURL: this.baseURL,
            model: resolvedModel,
            requestModel: request.model,
            toolsSummary: summarizeToolNames(tools),
            payload: {
              messages,
              tools,
              stream: false,
              tool_choice: tools && tools.length > 0 ? 'auto' : undefined,
              ...this.config,
              model: resolvedModel,
            },
          },
          null,
          2,
        ),
      );
    }
    const completion = await this.client.chat.completions.create({
      messages,
      stream: false,
      tools,
      tool_choice: tools && tools.length > 0 ? 'auto' : undefined,
      ...this.config,
      model: resolvedModel,
    });
    if (shouldDebugApi(userPromptId)) {
      debugLogger.log(
        '[custom-llm-api-response]',
        JSON.stringify(
          {
            promptId: userPromptId,
            response: completion,
          },
          null,
          2,
        ),
      );
    }
    return ModelConverter.toGeminiResponse(completion);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const messages = ModelConverter.toOpenAIMessages(request);
    const tools = extractToolFunctions(request.config) ?? [];
    const resolvedModel = this.getResolvedModel(request.model);
    if (shouldDebugApi(userPromptId)) {
      debugLogger.log(
        '[custom-llm-api-request]',
        JSON.stringify(
          {
            promptId: userPromptId,
            baseURL: this.baseURL,
            model: resolvedModel,
            requestModel: request.model,
            toolsSummary: summarizeToolNames(tools),
            payload: {
              messages,
              tools,
              stream: true,
              tool_choice: tools.length > 0 ? 'auto' : undefined,
              ...this.config,
              model: resolvedModel,
            },
          },
          null,
          2,
        ),
      );
    }
    const stream = await this.client.chat.completions.create({
      messages,
      stream: true,
      tools,
      tool_choice: tools.length > 0 ? 'auto' : undefined,
      ...this.config,
      model: resolvedModel,
    });
    const toolCallMap: ToolCallMap = new Map();

    return (async function* (): AsyncGenerator<GenerateContentResponse> {
      let sawAnyToolCallChunk = false;
      let sawAnyContentChunk = false;
      for await (const chunk of stream) {
        const hasToolCall = chunk.choices?.some((choice) =>
          Array.isArray(choice.delta?.tool_calls),
        );
        const hasContent = chunk.choices?.some(
          (choice) => typeof choice.delta?.content === 'string',
        );
        sawAnyToolCallChunk ||= hasToolCall;
        sawAnyContentChunk ||= hasContent;
        if (shouldDebugApiChunks(userPromptId)) {
          debugLogger.log(
            '[custom-llm-api-response-chunk]',
            JSON.stringify(
              {
                promptId: userPromptId,
                hasToolCall,
                hasContent,
                chunk,
              },
              null,
              2,
            ),
          );
        }
        if (shouldDebugApi(userPromptId) && hasToolCall) {
          debugLogger.log(
            '[custom-llm-tool-call-chunk]',
            JSON.stringify(
              {
                promptId: userPromptId,
                chunk,
              },
              null,
              2,
            ),
          );
        }
        const { response } = ModelConverter.processStreamChunk(
          chunk,
          toolCallMap,
        );
        if (shouldDebugApi(userPromptId) && response?.functionCalls?.length) {
          debugLogger.log(
            '[custom-llm-tool-call-response]',
            JSON.stringify(
              {
                promptId: userPromptId,
                functionCalls: response.functionCalls,
              },
              null,
              2,
            ),
          );
        }
        if (response) {
          yield response;
        }
      }
      if (shouldDebugApi(userPromptId)) {
        debugLogger.log(
          '[custom-llm-stream-summary]',
          JSON.stringify(
            {
              promptId: userPromptId,
              sawAnyToolCallChunk,
              sawAnyContentChunk,
              toolCallMapSize: toolCallMap.size,
              toolCallMap: Array.from(toolCallMap.entries()),
            },
            null,
            2,
          ),
        );
      }
    })();
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    const contents = normalizeContents(request.contents ?? []);
    const text = contents
      .flatMap(
        (content) =>
          content.parts
            ?.map((part: unknown) => {
              if (typeof part === 'string') {
                return part;
              }
              if (typeof part === 'object' && part !== null) {
                const typedPart = part as {
                  text?: unknown;
                  inlineData?: { data?: unknown };
                };
                if (typeof typedPart.text === 'string') {
                  return typedPart.text;
                }
                if (typeof typedPart.inlineData?.data === 'string') {
                  return typedPart.inlineData.data;
                }
              }
              return '';
            })
            .filter(Boolean) ?? [],
      )
      .join(' ');

    const englishWords = (text.match(/[a-zA-Z]+[']?[a-zA-Z]*/g) ?? []).length;
    const chineseChars = (text.match(/[\u4e00-\u9fff]/g) ?? []).length;
    const numbers = (text.match(/\b\d+\b/g) ?? []).length;
    const punctuations =
      text.match(/[.,!?;:"'(){}[\]<>@#$%^&*\-_+=~`|\\/]/g)?.length ?? 0;
    const spaces = Math.ceil((text.match(/\s+/g) ?? []).length / 5);

    const totalTokens = Math.ceil(
      englishWords * 1.2 +
        chineseChars * 1 +
        numbers * 0.8 +
        punctuations * 0.5 +
        spaces,
    );

    return {
      totalTokens,
    };
  }

  async embedContent(
    _request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    throw new Error('Embedding is not supported for custom LLM providers.');
  }

  private getResolvedModel(requestModel: string): string {
    if (!this.modelName) {
      throw new Error(
        `CUSTOM_LLM_MODEL_NAME is required but missing. Received request model: ${requestModel}`,
      );
    }
    return this.modelName;
  }
}

function shouldDebugApi(promptId: string): boolean {
  return (
    process.env['GEMINI_DEBUG_API'] === '1' &&
    (process.env['GEMINI_DEBUG_API_ALL'] === '1' ||
      /########\d+$/.test(promptId))
  );
}

function shouldDebugApiChunks(promptId: string): boolean {
  return (
    shouldDebugApi(promptId) && process.env['GEMINI_DEBUG_API_CHUNKS'] === '1'
  );
}

function summarizeToolNames(
  tools: OpenAI.Chat.Completions.ChatCompletionTool[] | undefined,
): { count: number; names: string[] } {
  if (!tools || tools.length === 0) {
    return { count: 0, names: [] };
  }
  return {
    count: tools.length,
    names: tools
      .map((tool) => {
        if ('function' in tool && tool.function) {
          return tool.function.name;
        }
        if ('custom' in tool && tool.custom) {
          return tool.custom.name;
        }
        return '';
      })
      .filter((name) => name.length > 0),
  };
}
