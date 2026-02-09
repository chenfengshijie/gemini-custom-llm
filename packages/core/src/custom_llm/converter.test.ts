/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect } from 'vitest';
import { ModelConverter } from './converter.js';

describe('custom_llm ModelConverter.processStreamChunk', () => {
  it('flushes tool calls when finish_reason=tool_calls arrives in a later chunk', () => {
    const toolCallMap = new Map();

    const first = ModelConverter.processStreamChunk(
      {
        choices: [
          {
            index: 0,
            delta: {
              role: 'assistant',
              content: '',
              tool_calls: [
                {
                  id: 'call_123',
                  index: 0,
                  type: 'function',
                  function: { name: 'list_directory', arguments: '' },
                },
              ],
            },
          },
        ],
      } as any,
      toolCallMap,
    );
    expect(first.response).toBeUndefined();

    const second = ModelConverter.processStreamChunk(
      {
        choices: [
          {
            index: 0,
            delta: {
              role: 'assistant',
              content: '',
              tool_calls: [
                {
                  index: 0,
                  function: { arguments: '{"dir_path":"/tmp"}' },
                },
              ],
            },
          },
        ],
      } as any,
      toolCallMap,
    );
    expect(second.response).toBeUndefined();

    const third = ModelConverter.processStreamChunk(
      {
        choices: [
          {
            index: 0,
            finish_reason: 'tool_calls',
            delta: {
              role: 'assistant',
              content: '',
            },
          },
        ],
      } as any,
      toolCallMap,
    );

    expect(third.response?.functionCalls).toHaveLength(1);
    expect(third.response?.functionCalls?.[0]).toMatchObject({
      id: 'call_123',
      name: 'list_directory',
      args: { dir_path: '/tmp' },
    });
    expect(toolCallMap.size).toBe(0);
  });
});

