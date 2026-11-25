import type {
  MessageStreamEvent,
  MessageUsage,
  ServerToolBlock,
  StopReason,
  TextCitation,
} from '../types.js'

/** 统一构造用量；没有缓存信息时补 0，界面按字段存在与否展示 */
export const usageOf = (input: number, output: number, cached = 0): MessageUsage => ({
  input_tokens: Math.max(0, input - cached),
  cache_creation_input_tokens: 0,
  cache_read_input_tokens: cached,
  output_tokens: output,
})

/**
 * 把各协议的增量文本 / 思考片段拼成规范的块事件。
 * OpenAI、Responses、Gemini 都不给块编号，这里按内容类别编块：同类增量共用一个打开的块，
 * 切换类别时先关闭上一个块，与 Anthropic 的事件顺序保持一致。
 */
export class BlockStream {
  private index = -1
  private open: { index: number; kind: 'text' | 'thinking' } | null = null;

  private *start(kind: 'text' | 'thinking'): Generator<MessageStreamEvent> {
    yield* this.close()
    this.index += 1
    this.open = { index: this.index, kind }
    yield {
      type: 'content_block_start',
      index: this.index,
      content_block:
        kind === 'text'
          ? { type: 'text', text: '' }
          : { type: 'thinking', thinking: '', signature: '' },
    }
  }

  private *delta(kind: 'text' | 'thinking', value: string): Generator<MessageStreamEvent> {
    if (!value) {
      return
    }
    if (this.open?.kind !== kind) {
      yield* this.start(kind)
    }
    const index = this.open?.index ?? 0
    yield {
      type: 'content_block_delta',
      index,
      delta:
        kind === 'text'
          ? { type: 'text_delta', text: value }
          : { type: 'thinking_delta', thinking: value },
    }
  }

  *text(value: string | null | undefined): Generator<MessageStreamEvent> {
    if (value) {
      yield* this.delta('text', value)
    }
  }

  *thinking(value: string | null | undefined): Generator<MessageStreamEvent> {
    if (value) {
      yield* this.delta('thinking', value)
    }
  }

  /** 追加一个完整的服务端工具块：先结束当前文本 / 思考块，再按顺序分配新索引 */
  *tool(block: ServerToolBlock): Generator<MessageStreamEvent> {
    yield* this.close()
    this.index += 1
    const index = this.index
    yield { type: 'content_block_start', index, content_block: block }
    yield { type: 'content_block_stop', index }
  }

  /** 给当前打开的文本块补一条引用；没有打开的文本块时忽略，避免产生悬空引用 */
  *citation(citation: TextCitation): Generator<MessageStreamEvent> {
    if (this.open?.kind !== 'text') {
      return
    }
    const { index } = this.open
    yield {
      type: 'content_block_delta',
      index,
      delta: { type: 'citations_delta', citation },
    }
  }

  *close(): Generator<MessageStreamEvent> {
    if (!this.open) {
      return
    }
    const { index } = this.open
    this.open = null
    yield { type: 'content_block_stop', index }
  }

  /** 关闭打开的块并结束本轮；usage 在没有时不下发 */
  *finish(
    stopReason: StopReason,
    usage?: MessageUsage,
    title?: string
  ): Generator<MessageStreamEvent> {
    yield* this.close()
    yield {
      type: 'message_delta',
      delta: {
        stop_reason: stopReason,
        stop_sequence: null,
        ...(title ? { title } : {}),
      },
      usage,
    }
    yield { type: 'message_stop' }
  }
}

/** 各协议的结束原因归一化为应用内的四种 */
export const mapFinishReason = (reason: string | null | undefined): StopReason | null => {
  switch (reason) {
    case undefined:
    case null:
    case '':
      return null
    case 'stop':
    case 'end_turn':
    case 'stop_sequence':
    case 'STOP':
      return 'end_turn'
    case 'length':
    case 'max_tokens':
    case 'MAX_TOKENS':
    case 'model_length':
      return 'max_tokens'
    case 'content_filter':
    case 'refusal':
    case 'SAFETY':
    case 'RECITATION':
    case 'BLOCKLIST':
    case 'PROHIBITED_CONTENT':
    case 'SPII':
      return 'refusal'
    case 'tool_calls':
    case 'tool_use':
      return 'tool_use'
    default:
      return 'end_turn'
  }
}
