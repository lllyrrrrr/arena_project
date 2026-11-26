
===== t2 =====
交集: 基础榜对照 n=35 / 风格榜对照 n=35 / 变动对照 n=35
未匹配(9): ['codestral-2405', 'dbrx-next', 'deepseek-v2-api-0628', 'eureka-chatbot', 'gemini-1.5-pro-exp-0801', 'reka-core-20240501', 'reka-flash-preview-20240611', 'yi-large', 'yi-large-preview']
  本研究基础榜 vs 官方full             rho=0.958  p=2.02e-19
  本研究控风格榜 vs 官方style           rho=0.958  p=1.63e-19
  名次变动向量(delta)                rho=0.720  p=1.10e-06
  变动方向一致率: 70% (n=27)
  本研究|delta|>=3: 方向一致率 92% (n=12)  delta rho=0.920 p=2.24e-05
  对称大变动(|d_ours|>=3或|d_off|>=3): 方向一致率 87% (n=15)  delta rho=0.818 p=1.92e-04
  变动最大的共同模型 (交集内重排, +=控风格后上升):
  OK gpt-4-0613                           本研究+12 官方+5
  OK gpt-3.5-turbo-0125                   本研究+10 官方+4
  OK yi-1.5-34b-chat                      本研究-8 官方-4
  OK gpt-4o-mini-2024-07-18               本研究-6 官方-5
  OK llama-3-8b-instruct                  本研究-6 官方-1
  OK deepseek-coder-v2                    本研究+5 官方+2
  OK nemotron-4-340b-instruct             本研究-5 官方-1
  OK claude-3-sonnet-20240229             本研究+4 官方+3
  OK claude-3-5-sonnet-20240620           本研究+3 官方+1
  XX command-r                            本研究-3 官方+1
  OK dbrx-instruct-preview                本研究+3 官方+1
  OK gemma-2-9b-it                        本研究-3 官方-4

===== t3 =====
交集: 基础榜对照 n=41 / 风格榜对照 n=41 / 变动对照 n=41
未匹配(7): ['amazon-nova-experimental-chat-05-14', 'gemini-2.5-flash-lite-preview-06-17-thinking', 'gemini-2.5-flash-preview-04-17', 'gemini-2.5-pro-preview-05-06', 'llama-4-maverick-03-26-experimental', 'qwen-max-2025-01-25', 'qwen3-235b-a22b-instruct-2507']
  本研究基础榜 vs 官方full             rho=0.953  p=8.62e-22
  本研究控风格榜 vs 官方style           rho=0.956  p=2.70e-22
  名次变动向量(delta)                rho=0.914  p=7.82e-17
  变动方向一致率: 92% (n=39)
  本研究|delta|>=3: 方向一致率 94% (n=34)  delta rho=0.879 p=8.37e-12
  对称大变动(|d_ours|>=3或|d_off|>=3): 方向一致率 94% (n=35)  delta rho=0.877 p=4.78e-12
  变动最大的共同模型 (交集内重排, +=控风格后上升):
  OK claude-3-7-sonnet-20250219-thinking- 本研究+17 官方+12
  OK claude-3-7-sonnet-20250219           本研究+13 官方+10
  OK claude-3-5-sonnet-20241022           本研究+12 官方+10
  OK grok-3-preview-02-24                 本研究-12 官方-7
  OK kimi-k2-0711-preview                 本研究+12 官方+7
  OK claude-opus-4-20250514               本研究+11 官方+10
  OK claude-sonnet-4-20250514-thinking-32 本研究+11 官方+12
  OK gemma-3-27b-it                       本研究-11 官方-5
  OK gemma-3n-e4b-it                      本研究-10 官方-6
  OK claude-opus-4-20250514-thinking-16k  本研究+9 官方+9
  OK claude-sonnet-4-20250514             本研究+9 官方+8
  OK grok-3-mini-beta                     本研究-9 官方-8