"""Build a local blinded audio/video annotation review sheet from frozen clips."""

# ruff: noqa: E501 -- Inline HTML and JavaScript are preserved as a single template.

import argparse
import hashlib
import json
from pathlib import Path

from scripts.train_source_unimodal_cv import digest

SAMPLE_SHA = "9d7df2419a4b3b63b23d1da94f571cf2ba29f3492b433caad0d530c756531126"


def blinded_rows(sample):
    rows = sample["samples"]
    if len(rows) != 16 or len({r["record_id"] for r in rows}) != 16:
        raise ValueError("unexpected sample coverage")
    ordered = sorted(
        rows,
        key=lambda r: hashlib.sha256(
            f"review-blind-order-v1:17:{r['record_id']}".encode()
        ).digest(),
    )
    result = []
    for index, row in enumerate(ordered, 1):
        path = Path(row["media_path"])
        if digest(path) != row["media_sha256"]:
            raise ValueError("media hash mismatch")
        start, end = row["audio_interval_us"]
        if not 0 <= start < end:
            raise ValueError("invalid event interval")
        result.append(
            {
                "id": f"R{index:02d}",
                "record_id": row["record_id"],
                "url": path.resolve().as_uri(),
                "start": start / 1e6,
                "end": end / 1e6,
            }
        )
    return result


def render(rows):
    data = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    return """<!doctype html><html lang="ja"><meta charset="utf-8">
<title>音声・映像の注釈確認</title><style>
body{font:16px system-ui;max-width:960px;margin:2rem auto;padding:0 1rem;line-height:1.6}
video{width:100%;max-height:65vh;background:#111} label{display:block;margin:.6rem 0}
select,textarea,button{font:inherit;padding:.4rem}textarea{width:98%;height:5rem}
.warning{background:#fff3cd;padding:1rem}button{margin:.3rem}
</style><h1>音声・映像の注釈確認</h1>
<p class="warning">元のラベルとモデル予測は表示していません。音をオンにし、指定された時間帯と前後を確認してください。判断できない場合は「不明」を選んでください。この画面は学習データを変更しません。</p>
<p><span id="position"></span>　<span id="time"></span></p>
<video id="video" controls preload="metadata"></video>
<p><button id="replay">イベント直前から再生</button> 元動画全体をシークできます。</p>
<label>指定区間で音を聞き取れますか？ <select id="audible"><option value="">未回答</option><option>はい</option><option>いいえ</option><option>不明</option></select></label>
<label>音の発生に関係する物体を映像から特定できますか？ <select id="object"><option value="">未回答</option><option>はい</option><option>いいえ</option><option>不明</option></select></label>
<label>音が生じる接触点・発生箇所は見えますか？ <select id="contact"><option value="">未回答</option><option>はい</option><option>いいえ</option><option>不明</option></select></label>
<label>複数の物体が関係すると判断できますか？ <select id="multiple"><option value="">未回答</option><option>はい</option><option>いいえ</option><option>不明</option></select></label>
<label>根拠や迷った点<textarea id="notes"></textarea></label>
<button id="prev">前へ</button><button id="next">次へ</button><button id="export">回答をJSONで保存</button>
<p id="status"></p><script>
const clips = DATA_PLACEHOLDER;
const key = "source-semantics-review-v1-9d7df241";
let index = 0, answers = JSON.parse(localStorage.getItem(key) || "{}");
const fields = ["audible", "object", "contact", "multiple", "notes"];
function persist(){const x={record_id:clips[index].record_id};fields.forEach(f=>x[f]=document.getElementById(f).value);answers[clips[index].id]=x;localStorage.setItem(key,JSON.stringify(answers));}
function show(){let x=clips[index], video=document.getElementById("video");video.src=x.url;video.onloadedmetadata=()=>{video.currentTime=Math.max(0,x.start-1)};document.getElementById("position").textContent=x.id+" / "+clips.length;document.getElementById("time").textContent="対象: "+x.start.toFixed(3)+"〜"+x.end.toFixed(3)+" 秒";let a=answers[x.id]||{};fields.forEach(f=>document.getElementById(f).value=a[f]||"");document.getElementById("status").textContent="回答はこのブラウザー内に保存されます。最後にJSONを書き出してください。";}
fields.forEach(f=>document.getElementById(f).addEventListener("change",persist));
document.getElementById("prev").onclick=()=>{persist();index=Math.max(0,index-1);show()};
document.getElementById("next").onclick=()=>{persist();index=Math.min(clips.length-1,index+1);show()};
document.getElementById("replay").onclick=()=>{let v=document.getElementById("video");v.currentTime=Math.max(0,clips[index].start-1);v.play()};
document.getElementById("export").onclick=()=>{persist();let a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify({sample_sha256:"SAMPLE_SHA_PLACEHOLDER",answers},null,2)],{type:"application/json"}));a.download="source-semantics-review.json";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
show();</script></html>""".replace("DATA_PLACEHOLDER", data).replace(
        "SAMPLE_SHA_PLACEHOLDER", SAMPLE_SHA
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or digest(args.sample) != SAMPLE_SHA:
        raise ValueError("output exists or frozen sample hash mismatch")
    rows = blinded_rows(json.loads(args.sample.read_bytes()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        out.write(render(rows))
    print("review sheet SHA-256:", digest(args.output))


if __name__ == "__main__":
    main()
