/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  GenerateContentResponse,
  type FunctionCall,
  type Part,
  type GenerateContentParameters,
} from '@google/genai';
import type OpenAI from 'openai';
import {
  normalizeContents,
  isValidFunctionCall,
  isValidFunctionResponse,
} from './util.js';
import type { ToolCallMap } from './types.js';

type OpenAIToolCall =
  | {
      id?: string;
      type: 'function';
      function: { name: string; arguments: string };
    }
  | {
      id?: string;
      type: 'custom';
      custom: { name: string; input: string };
    };

type OpenAIStreamToolCall = {
  index: number;
  id?: string;
  type?: string;
  function?: { name?: string; arguments?: string };
};

export class ModelConverter {
  /**
   * Convert Gemini content to OpenAI messages.
   */
  static toOpenAIMessages(
    request: GenerateContentParameters,
  ): OpenAI.Chat.Completions.ChatCompletionMessageParam[] {
    const { contents, config } = request;
    const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [];
    const systemInstruction = config?.systemInstruction;
    if (typeof systemInstruction === 'string' && systemInstruction.length > 0) {
      messages.push({
        role: 'system',
        content: systemInstruction,
      });
    }

    const contentsArray = normalizeContents(contents);
    for (const content of contentsArray) {
      const role =
        content.role === 'model' ? 'assistant' : (content.role as string);
      const parts = content.parts ?? [];
      this.processTextParts(parts, role, messages);
      this.processFunctionResponseParts(parts, messages);
      this.processFunctionCallParts(parts, messages);
    }
    return messages;
  }

  private static processTextParts(
    parts: Part[],
    role: string,
    messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[],
  ): void {
    const textParts = parts.filter(
      (part): part is { text: string } =>
        typeof part === 'object' && part !== null && 'text' in part,
    );
    if (textParts.length === 0) {
      return;
    }
    const text = textParts.map((part) => part.text).join('\n');
    if (role === 'user') {
      messages.push({ role: 'user', content: text });
    } else if (role === 'system') {
      messages.push({ role: 'system', content: text });
    } else if (role === 'assistant') {
      messages.push({ role: 'assistant', content: text });
    }
  }

  private static processFunctionResponseParts(
    parts: Part[],
    messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[],
  ): void {
    const frParts = parts.filter(isValidFunctionResponse);
    if (frParts.length === 0) {
      return;
    }
    for (const part of frParts) {
      messages.push({
        tool_call_id: part.functionResponse.id,
        role: 'tool',
        content: part.functionResponse.response.error
          ? `Error: ${part.functionResponse.response.error}`
          : (part.functionResponse.response.output ?? ''),
      });
    }
  }

  private static processFunctionCallParts(
    parts: Part[],
    messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[],
  ): void {
    const fcParts = parts.filter(isValidFunctionCall);
    if (fcParts.length === 0) {
      return;
    }
    messages.push({
      role: 'assistant',
      content: null,
      tool_calls: fcParts.map((part) => ({
        id: part.functionCall.id,
        type: 'function',
        function: {
          name: part.functionCall.name,
          arguments: JSON.stringify(part.functionCall.args),
        },
      })),
    });
  }

  /**
   * Convert OpenAI response to Gemini response.
   */
  static toGeminiResponse(
    response: OpenAI.Chat.Completions.ChatCompletion,
  ): GenerateContentResponse {
    const choice = response.choices[0];
    const res = new GenerateContentResponse();

    if (choice.message.content) {
      res.candidates = [
        {
          content: {
            parts: [{ text: choice.message.content }],
            role: 'model',
          },
          index: 0,
          safetyRatings: [],
        },
      ];
    } else if (choice.message.tool_calls) {
      const toolCalls = choice.message.tool_calls as OpenAIToolCall[];
      const functionCalls: FunctionCall[] = [];
      const parts: Part[] = [];
      res.candidates = [
        {
          content: {
            parts,
          },
          index: 0,
          safetyRatings: [],
        },
      ];
      attachFunctionCalls(res, functionCalls);
      for (const toolCall of toolCalls) {
        if (toolCall.type === 'function') {
          const functionCall: FunctionCall = {
            id: toolCall.id,
            name: toolCall.function.name,
            args: safeJsonParse(toolCall.function.arguments),
          };
          functionCalls.push(functionCall);
          parts.push({ functionCall });
          continue;
        }
        if (toolCall.type === 'custom') {
          const functionCall: FunctionCall = {
            id: toolCall.id,
            name: toolCall.custom.name,
            args: { input: toolCall.custom.input },
          };
          functionCalls.push(functionCall);
          parts.push({ functionCall });
        }
      }
    }

    res.usageMetadata = {
      promptTokenCount: response.usage?.prompt_tokens ?? 0,
      candidatesTokenCount: response.usage?.completion_tokens ?? 0,
      totalTokenCount: response.usage?.total_tokens ?? 0,
    };

    return res;
  }

  /**
   * Convert streaming chunks to Gemini responses while tracking tool calls.
   */
  static processStreamChunk(
    chunk: OpenAI.Chat.Completions.ChatCompletionChunk,
    toolCallMap: ToolCallMap,
  ): { response?: GenerateContentResponse } {
    const delta = chunk.choices[0]?.delta;
    const finishReason = chunk.choices[0]?.finish_reason;
    if (!delta) {
      return {};
    }

    const rawContent: unknown = (delta as { content?: unknown }).content;

    if (typeof rawContent === 'string' && rawContent.length > 0) {
      const response = new GenerateContentResponse();
      response.candidates = [
        {
          content: {
            role: 'model',
            parts: [{ text: rawContent }],
          },
          index: 0,
          safetyRatings: [],
        },
      ];
      return { response };
    }

    if (Array.isArray(rawContent) && rawContent.length > 0) {
      const first = rawContent[0];
      if (
        typeof first === 'object' &&
        first !== null &&
        'type' in first &&
        'text' in first &&
        (first as { type?: unknown; text?: unknown }).type === 'text' &&
        typeof (first as { text?: unknown }).text === 'string' &&
        (first as { text: string }).text.length > 0
      ) {
        const response = new GenerateContentResponse();
        response.candidates = [
          {
            content: {
              role: 'model',
              parts: [{ text: (first as { text: string }).text }],
            },
            index: 0,
            safetyRatings: [],
          },
        ];
        return { response };
      }
    }

    const toolCalls = delta.tool_calls as OpenAIStreamToolCall[] | undefined;
    if (toolCalls && toolCalls.length > 0) {
      for (const call of toolCalls) {
        if (call.type && call.type !== 'function') {
          continue;
        }
        const current = toolCallMap.get(call.index) ?? {
          id: undefined,
          name: '',
          arguments: '',
        };

        if (call.id) {
          current.id = call.id;
        }

        if (call.function?.name) {
          current.name = call.function.name;
        }

        if (call.function?.arguments) {
          current.arguments += call.function.arguments;
        }

        toolCallMap.set(call.index, current);
      }
    }

    if (finishReason === 'tool_calls' && toolCallMap.size > 0) {
      return { response: flushToolCallMap(toolCallMap) };
    }

    return {};
  }
}

function flushToolCallMap(toolCallMap: ToolCallMap): GenerateContentResponse {
  const functionCalls: FunctionCall[] = [];
  const parts: Part[] = [];
  for (const [, data] of Array.from(toolCallMap.entries()).sort(
    ([left], [right]) => left - right,
  )) {
    if (!data.name) {
      continue;
    }
    const functionCall: FunctionCall = {
      id: data.id,
      name: data.name,
      args: safeJsonParse(data.arguments),
    };
    functionCalls.push(functionCall);
    parts.push({ functionCall });
  }

  toolCallMap.clear();

  const response = new GenerateContentResponse();
  response.candidates = [
    {
      content: {
        role: 'model',
        parts,
      },
      index: 0,
      safetyRatings: [],
    },
  ];
  attachFunctionCalls(response, functionCalls);
  return response;
}

function attachFunctionCalls(
  response: GenerateContentResponse,
  functionCalls: FunctionCall[],
): void {
  if (functionCalls.length === 0) {
    return;
  }
  Object.defineProperty(response, 'functionCalls', {
    value: functionCalls,
    enumerable: true,
    configurable: true,
  });
}

function safeJsonParse(raw: string): Record<string, unknown> {
  try {
    return JSON.parse(raw);
  } catch {
    return {};
  }
}
