import json
import re
import sys

REFUSAL = re.compile(
    r"(do not|don.{0,6}t) have enough information|cannot be answered|not (enough|sufficient) information|passages? (do|does) not",
    re.I,
)


def rescore(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
        ans = [r for r in rows if r["answerable"]]
        unans = [r for r in rows if not r["answerable"]]
    for r in rows:
        r["refused"] = bool(REFUSAL.search(r["answer"]))
    fr = [r for r in ans if r["refused"]]
    print(path)
    print(
        f"  false refusals:   {len(fr)}/{len(ans)} = {len(fr) / len(ans):.2f}"
        f"  (source retrieved in {sum(r['retrieved'] for r in fr)})"
    )
    print(f"  correct refusals: {sum(r['refused'] for r in unans)}/{len(unans)}")
    if "correct" in rows[0]:
        done = [r for r in ans if not r["refused"]]
        print(f"  correct (all):    {sum(bool(r.get('correct')) for r in ans) / len(ans):.2f}")
        print(
            f"  correct (answered only): {sum(bool(r.get('correct')) for r in done) / len(done):.2f} (n={len(done)})"
        )
        fa = [r for r in rows if not r["refused"] and r.get("faithful") is not None]
        print(
            f"  faithful (answered): {sum(bool(r['faithful']) for r in fa) / len(fa):.2f} (n={len(fa)})"
        )


for p in sys.argv[1:]:
    rescore(p)
