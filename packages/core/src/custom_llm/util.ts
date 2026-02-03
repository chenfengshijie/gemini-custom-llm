/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import OpenAI from 'openai';
import type {
  Part,
  Content,
  ContentListUnion,
  GenerateContentConfig,
} from '@google/genai';

export function isValidFunctionCall(part: Part): part is {
  functionCall: { name: string; args: Record<string, unknown>; id: string };
} {
  if (typeof part !== 'object' || part === null) {
    return false;
  }
  const { functionCall } = part as { functionCall?: unknown };
  if (!functionCall || typeof functionCall !== 'object') {
    return false;
  }
  const { name, args, id } = functionCall as {
    name?: unknown;
    args?: unknown;
    id?: unknown;
  };
  return typeof name === 'string' && args !== undefined && typeof id === 'string';
}

/**
 * Type guard for function response parts.
 */
export function isValidFunctionResponse(part: Part): part is {
  functionResponse: {
    id: string;
    name: string;
    response: { output?: string; error?: string };
  };
} {
  if (typeof part !== 'object' || part === null) {
    return false;
  }
  const { functionResponse } = part as { functionResponse?: unknown };
  if (!functionResponse || typeof functionResponse !== 'object') {
    return false;
  }
  const { id, name, response } = functionResponse as {
    id?: unknown;
    name?: unknown;
    response?: unknown;
  };
  if (typeof id !== 'string' || typeof name !== 'string') {
    return false;
  }
  if (!response || typeof response !== 'object') {
    return false;
  }
  return true;
}

/**
 * Extracts the answer portion from an LLM output that may include
 * thinking tags such as `<think>...</think>`.
 */
export function extractAnswer(text: string): string {
  const startTags = ['<think>', '<thinking>'];
  const endTags = ['</think>', '</thinking>'];
  for (let i = 0; i < startTags.length; i++) {
    const start = startTags[i];
    const end = endTags[i];
    if (text.includes(start) && text.includes(end)) {
      const partsBefore = text.split(start);
      const partsAfter = partsBefore[1].split(end);
      return (partsBefore[0].trim() + ' ' + partsAfter[1].trim()).trim();
    }
  }
  return text;
}

/**
 * Attempts to extract JSON from LLM output.
 */
export function extractJsonFromLLMOutput(output: string): unknown {
  if (output.trim().startsWith('<think')) {
    output = extractAnswer(output);
  }
  try {
    return JSON.parse(output);
  } catch {
    // Fall through to fenced code block detection.
  }
  const jsonStart = output.indexOf('```json');
  const jsonEnd = output.lastIndexOf('```');
  if (jsonStart !== -1 && jsonEnd !== -1) {
    const jsonString = output.substring(jsonStart + 7, jsonEnd);
    try {
      return JSON.parse(jsonString);
    } catch (error) {
      console.error('Failed to parse JSON from fenced block:', {
        error,
        llmResponse: output,
      });
    }
  } else {
    console.error('LLM output not in expected JSON format:', output);
  }
  return undefined;
}

/**
 * Convert any `type` values in a JSON schema to lowercase to satisfy providers
 * that expect lowercase type declarations.
 */
function convertTypeValuesToLowerCase(obj: unknown): unknown {
  if (Array.isArray(obj)) {
    return obj.map((item) => convertTypeValuesToLowerCase(item));
  }
  if (obj !== null && typeof obj === 'object') {
    const newObj: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(obj)) {
      if (key === 'minLength' || key === 'minItems') {
        continue;
      }
      if (key === 'type' && typeof value === 'string') {
        newObj[key] = value.toLowerCase();
      } else {
        newObj[key] = convertTypeValuesToLowerCase(value);
      }
    }
    return newObj;
  }
  return obj;
}

/**
 * Converts Gemini tool function declarations to OpenAI-compatible tool array format.
 */
export function extractToolFunctions(
  requestConfig: GenerateContentConfig | undefined,
): OpenAI.Chat.Completions.ChatCompletionTool[] | undefined {
  if (!requestConfig?.tools) return undefined;
  const result: OpenAI.Chat.Completions.ChatCompletionTool[] = [];
  for (const tool of requestConfig.tools) {
    if ('functionDeclarations' in tool) {
      for (const func of tool.functionDeclarations ?? []) {
        result.push({
          type: 'function',
          function: {
            name: func.name ?? '',
            description: func.description ?? '',
            parameters: convertTypeValuesToLowerCase(
              func.parameters,
            ) as Record<string, unknown>,
          },
        });
      }
    }
  }
  return result.length > 0 ? result : undefined;
}

/**
 * Normalize contents to an array of `Content` objects.
 */
export function normalizeContents(contents: ContentListUnion): Content[] {
  if (Array.isArray(contents)) {
    return contents.map((item) => {
      if (typeof item === 'string') {
        return { role: 'user', parts: [{ text: item }] };
      }
      if (typeof item === 'object' && item !== null && 'parts' in item) {
        return item as Content;
      }
      return {
        role: 'user',
        parts: [item as Part],
      };
    });
  }
  if (typeof contents === 'string') {
    return [
      {
        role: 'user',
        parts: [{ text: contents }],
      },
    ];
  }
  if (typeof contents === 'object' && contents !== null && 'parts' in contents) {
    return [contents as Content];
  }
  return [
    {
      role: 'user',
      parts: [contents as Part],
    },
  ];
}