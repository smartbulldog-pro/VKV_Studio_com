"""One conversation turn against the local Synapse test server.
Usage: python turn.py <conversation_json> <message_txt>
Appends the persona's message (read UTF-8 from message_txt) to the conversation,
calls the server, appends Synapse's reply, saves, and PRINTS the reply.
"""
import sys, json, urllib.request, os

conv_file, msg_file = sys.argv[1], sys.argv[2]
msg = open(msg_file, encoding="utf-8").read().strip()
conv = []
if os.path.exists(conv_file):
    try:
        conv = json.load(open(conv_file, encoding="utf-8"))
    except Exception:
        conv = []
conv.append({"role": "user", "content": msg})
req = urllib.request.Request(
    "http://127.0.0.1:8777/chat",
    data=json.dumps({"messages": conv}, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
reply = json.loads(urllib.request.urlopen(req, timeout=600).read().decode("utf-8")).get("reply", "")
conv.append({"role": "assistant", "content": reply})
os.makedirs(os.path.dirname(conv_file) or ".", exist_ok=True)
json.dump(conv, open(conv_file, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(reply)
