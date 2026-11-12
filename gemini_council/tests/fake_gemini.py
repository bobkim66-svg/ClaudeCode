"""가짜 Gemini CLI. 실제 CLI(0.68.0)의 입출력 규약을 흉내 낸다 (로그인 없이 백엔드 테스트용).

환경변수로 동작 선택:
  FAKE_GEMINI_MODE: ok | quota | quota_stdout | auth | hang | garbage | empty | downgrade | crash | noise
  FAKE_GEMINI_LOG:  호출마다 {"argv", "stdin", "cwd"} 를 JSON 한 줄로 추가 기록할 파일
  FAKE_GEMINI_END_AFTER: 사회자 호출이 이 횟수에 도달하면 응답 끝에 종료 토큰 추가
"""
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    argv = sys.argv[1:]
    stdin = sys.stdin.buffer.read().decode("utf-8")
    mode = os.environ.get("FAKE_GEMINI_MODE", "ok")
    model = argv[argv.index("-m") + 1] if "-m" in argv else "default"
    log = os.environ.get("FAKE_GEMINI_LOG")
    n_calls = 0
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"argv": argv, "stdin": stdin, "cwd": os.getcwd()}, ensure_ascii=False) + "\n")
        n_calls = len(Path(log).read_text(encoding="utf-8").splitlines())
    assert "-p" in argv and "-o" in argv, "헤드리스 JSON 모드로 호출되지 않음"
    assert "--yolo" not in argv and "-y" not in argv

    def err(type_, msg, code, rc, stream=sys.stderr):
        stream.write('Approval mode overridden to "default" because the current folder is not trusted.\n')
        stream.write(json.dumps({"session_id": "x", "error": {"type": type_, "message": msg, "code": code}},
                                indent=2) + "\n")
        return rc

    if mode == "auth":
        return err("Error", "Please set an Auth method in settings.json", 41, 41)
    if mode == "quota":  # POSIX 에서 429 → 173
        return err("TerminalQuotaError", "You have exhausted your daily quota.", 429, 173)
    if mode == "quota_stdout":
        return err("RetryableQuotaError", "Resource exhausted", 429, 173, stream=sys.stdout)
    if mode == "hang":
        time.sleep(60)
        return 0
    if mode == "crash":
        sys.stderr.write("TypeError: cannot read properties of undefined\n")
        return 1
    if mode == "garbage":
        print("not json at all")
        return 0
    if mode == "empty":
        print(json.dumps({"session_id": "x", "response": "   "}))
        return 0

    is_mod = "사회자의 차례" in stdin
    text = f"(1) 입장: 반박\n(2) 근거: 호출 {n_calls}, 모델 {model}, 한글 ✓\n(3) 제안: 수정안"
    end_after = os.environ.get("FAKE_GEMINI_END_AFTER")
    if is_mod and end_after and log:
        mod_calls = sum("사회자의 차례" in json.loads(ln)["stdin"]
                        for ln in Path(log).read_text(encoding="utf-8").splitlines())
        if mod_calls >= int(end_after):
            text += "\n\n**[DISCUSSION_COMPLETE]**"
    used = "gemini-3-flash" if mode == "downgrade" or model != "pro" else "gemini-3-pro-preview"
    models = {used: {"tokens": {"total": 1000 + n_calls}},
              "gemini-3.1-flash-lite": {"tokens": {"total": 50}}}  # 내부 보조 호출 흉내
    if mode == "noise":
        print("Loaded cached credentials.")
    print(json.dumps({"session_id": "x", "response": text, "stats": {"models": models}},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
