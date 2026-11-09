# WIP: MP scheduler lookup burst (handoff)

이 디렉터리는 `perf/mp-bound-pending-lookups` PR을 위한 **측정 도구와 초안**입니다.
`wip/mp-lookup-burst-bench` branch에만 있고, PR에는 들어가지 않습니다.

## 1. 현재 상태

| 항목 | 상태 |
|---|---|
| PR branch | `daegyu94/LMCache:perf/mp-bound-pending-lookups`, `upstream/dev`(cd29b563) 위 commit 2개 |
| commit 1 `63b78a16` | 기능: 소비되지 않은 lookup 수를 `max_num_seqs`로 제한 |
| commit 2 `9fd40d81` | 원인 설명 정정(설계 문서, connector 주석), connector + 실제 adapter test 추가. **PR 전에 commit 1로 squash** |
| unit test (vLLM 0.33) | 관련 16개 파일 413 통과 / 1 실패(`test_free_locks_request_transport[grpc]`, 변경 전부터 실패) |
| upstream PR | **아직 안 만듦** |
| PR 본문, explainer, commit 메시지 초안 | `pr/` 아래. **원인 설명과 측정표가 낡았으므로 다시 써야 함** (5장) |

## 2. 변경 내용 (코드)

- `LMCacheMPSchedulerAdapter(max_pending_lookups: int | None = None)`와 `can_submit_lookup(request_id)`.
  `len(self._lookups)`(제출했고 아직 `cleanup_lookup_result` 안 된 lookup)가 상한 미만이거나,
  그 요청이 이미 lookup을 가지고 있으면 True.
- `LMCacheMPConnector.get_num_new_matched_tokens`: 제출 직전에 확인, 자리가 없으면 `(None, True)`.
- connector는 `max_num_seqs`를 넘기고, `lmcache.mp.eager_prefetch`가 켜지면 `None`(상한 없음).
- 기본값 `None`은 vLLM 내장 fallback connector가 이 adapter를 인자 없이 만들기 때문.

## 3. 측정으로 확인한 원인 (중요: 처음 PR 설명과 다름)

처음 가설 "server의 BLOCKING pool에서 PING이 LOOKUP 뒤에 줄을 선다"는 **틀렸음**.

- server의 LOOKUP handler는 1건당 약 0.1ms(burst 8,192건 합계 약 1초).
- 별도 process의 외부 PING probe는 burst 중에도 최대 0.14초.
- vLLM scheduler process의 **모든 LMCache RPC는 process당 하나인 thread
  `mq-client-shared-loop`(`lmcache/v1/multiprocess/transport/zmq_impl/mq.py`, `ClientPollingLoop`)**
  를 지남: LOOKUP 송신, ack, QUERY_PREFETCH_STATUS, 응답 callback, **heartbeat PING**
  (heartbeat는 lookup과 같은 `req_client` 사용, `vllm_multi_process_adapter.py`의 `HeartbeatThread(req_client=client)`).
- 상한이 없으면 LOOKUP이 초당 2,800\~3,600건 몰리고, client 제출에서 server 처리까지
  지연이 최대 4\~6초로 커짐. heartbeat PING 왕복 6.7\~10초, 3번 중 1번 timeout으로 degraded mode.
- **주 효과는 실행 시작 지연**: 곧 실행될 요청의 lookup도 이 줄 뒤에서 기다림
  (256번째 TTFT 9초 → 16\~21초). degraded mode는 부수 효과.
- 상한 256이면 제출 속도 초당 약 512건, 지연 0.12\~0.25초, PING 0.01초 이내.
- `QUERY_PREFETCH_STATUS`는 lookup당 정확히 1번(반복 polling 아님). scheduler `schedule()`의
  가장 긴 step은 0.24초라 GIL 독점만으로는 설명 안 됨.
- **미확정**: 밀리는 위치가 client loop thread인지, server의 해당 연결 수신 대기열인지까지는
  분리하지 못함(둘 다 "같은 연결"이라는 결론은 같음). 확정하려면 TODO 6.

### vLLM 0.33 측정값 (RTX PRO 4000 Blackwell 1장, Qwen3-0.6B, max_num_seqs=256, N=512 prompts x 16)

`data/vllm033_summary.txt`가 원본 요약. 시간은 burst 시작부터 초.

| | 상한 없음 | 상한 256 |
|---|---|---|
| 256번째 TTFT | 17.8 / 21.0 / 16.3 | 9.0 / 9.8 / 9.1 |
| TTFT 중앙값 | 38.6 / 42.2 / 36.9 | 26.8 / 27.8 / 26.9 |
| makespan | 61.8 / 65.7 / 60.5 | 46.4 / 48.2 / 47.4 |
| heartbeat PING 최대 | 8.1 / 10.0(timeout) / 6.7 | ≤ 0.01 |
| waiting queue 최대 | 약 7,960\~7,990 | 약 7,580\~7,620 |

세 값은 `trace_none`의 burst 2개, `trace_s_none` 순서(상한 256은 `trace_cap256` 2개, `trace_s_cap256`).
그 밖에 상한 없음 burst 1개(`trace_q_none`)는 TTFT 중앙값 41.3, makespan 64.9, PING 9.93초(timeout 직전).
degraded mode: 상한 없음 burst 4개 중 1번, 상한 256 burst 4개 중 0번. vanilla vLLM 0.33 값은 아직 없음.

## 4. 외부 리뷰 대응 (판정 완료)

- P1-a "완료된 lookup도 자리를 차지": 부분적으로 맞음. FCFS에서 vLLM은 할당 실패나 budget 소진 시
  `break`하므로 뒤 요청이 먼저 실행될 수 없음. 상한 = 한 step의 최대 admission 수.
  완료된 lookup을 세는 것은 lock과 L1 prefetch 무한 증가를 막음. 코드 변경 없음, 문서에 반영함.
- P1-b "256이 안전하다는 근거 없음": 맞는 주의. 보장이 아니라 경험적 상한으로 서술(반영함).
  상한 옵션은 추가하지 않음(coding standard 10.3).
- P2 "상한이 adapter 내부에서 강제되지 않음": 부분적으로 맞음. submit에서 거절하면
  `check_lookup_result`가 0(miss)이 되므로 분리가 의도. 코드 변경 없음.
- test: connector + 실제 adapter로 LOOKUP 수가 상한을 넘지 않는 test 추가함.
- 동기는 GRPO 등 특정 workload에 기대지 말 것: "waiting queue는 `max_num_seqs`보다 훨씬
  깊어질 수 있고, LMCache는 polling되는 모든 waiting request에 LOOKUP을 보낸다".
- vLLM 0.33에는 `max_num_queued_reqs` 없음(근거로 쓰지 말 것). `max_num_active_seqs`는
  `<= max_num_seqs`로 검증되므로 상한 근거 유지.

## 5. 해야 할 일 (순서대로)

1. **환경**: 이 branch를 checkout(PR commit 2개 + 이 디렉터리 포함).
   `bash wip/mp_lookup_burst/bench/setup_venv.sh <venv>` (vLLM 0.33.0, torch 2.14, native ext in-place build).
   LMCache CI가 테스트하는 vLLM: `https://raw.githubusercontent.com/LMCache/LMCache/buildkite_latest_tested_vllm/latest_tested_vllm.txt`
   (작성 시점 0.33.0). 더 새 버전이면 그 버전으로.
2. **vanilla 기준값**: `VENV=<venv> bash bench/run_trace.sh vanilla vanilla 512 512 32`
3. **요청 수별 측정** (외부 리뷰 요청): prompts = 16 32 64 128 256 512 (N = x16 = 256\~8,192).
   `run_trace.sh none_sweep none 16 32 64 128 256 512`, `run_trace.sh cap_sweep keep 16 ...`,
   `run_trace.sh van_sweep vanilla 16 ...`. 각 2회 이상. 기록: waiting queue 최대, heartbeat PING p99/최대,
   256번째 TTFT, makespan, degraded. 어디서부터 문제가 생기는지(queue 깊이 기준)가 PR의 핵심 근거.
4. (선택) 상한 민감도: cap 64/128/512로 같은 측정. 옵션 추가 목적이 아니라 `max_num_seqs` 선택 근거용.
5. (선택, GPU 여러 장일 때) TP=2/4에서 heartbeat 영향. worker rank별 PING도 같은 경로를 쓰는지 확인.
6. (선택) 지연 위치 확정: `MessageQueueClient.process_outbound_task`에서 PING이 실제 socket으로
   나간 시각을 기록해 client 송신 대기 vs server 수신 대기 분리.
7. **PR 산출물 다시 쓰기**:
   - commit 2개를 1개로 squash. 메시지는 `pr/commit_msg_draft.txt`를 3장 원인으로 고쳐 사용.
   - `pr/pr_body.md`: 동기(3장), 원인, vLLM 0.33 측정표, 알려진 한계(PRIORITY, LoRA/EC skip,
     eager prefetch, TP>1 미측정, 상한은 보장 아님), "heartbeat를 별도 경로로 분리" 대안 언급.
   - `pr/pr-explainer.html`: 원인 설명과 그림 1(server pool 대기열 그림)을 client loop 기준으로 다시 그리기,
     측정표 교체, SHA와 줄 번호 갱신. 저장소의 `.claude/skills/create-pr-explainer/SKILL.md` 절차를 따름.
8. **검증과 제출**: 독립 리뷰 agent, `.claude/skills/pre-pr-check`, ruff 0.11.7 / isort 6.0.1 / codespell 2.4.1,
   관련 test(vLLM 0.33). `git push --force-with-lease`로 squash 반영 후
   `LMCache/LMCache`의 `dev`로 **draft** PR. explainer HTML은 GitHub 웹에서 직접 첨부,
   "모든 변경 줄 직접 확인" 체크박스는 제출자가 직접.

## 6. 측정 도구

| 파일 | 역할 |
|---|---|
| `bench/run_trace.sh` | server + vLLM 실행, burst, 계측 수집. 자기가 띄운 process group만 정리 |
| `bench/burst.py` | prompts x samples 요청을 한꺼번에 streaming 전송, TTFT 분포 출력 |
| `bench/site/sitecustomize.py` | 측정 전용. `LMC_CAP`로 상한 강제, `LMC_TRACE*`로 heartbeat, LOOKUP 제출, server PING/LOOKUP/QUERY, `schedule()` 시간 기록 |
| `bench/pingprobe.py`, `bench/qprobe.py` | 외부 PING(0.5초 간격), vLLM waiting queue gauge 수집 |
| `bench/analyze.py` | `python analyze.py runs/trace_*` 로 burst별 요약 |
| `bench/setup_venv.sh` | vLLM venv 생성과 LMCache build |

주의:
- L1 40 GB는 호스트 RAM을 씀. N=8,192 두 번이면 eviction 발생.
- `sitecustomize`는 `PYTHONPATH` 맨 앞에 있어야 하고, 각 process(server, vLLM API, EngineCore)에 적용됨.
- vanilla 모드에서도 client 쪽 hook(`schedule()` 시간)은 기록됨.
