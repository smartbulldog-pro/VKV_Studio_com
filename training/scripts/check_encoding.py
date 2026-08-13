"""Check encoding of existing dataset and fix if needed"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

path = r'c:\projects\VKVstudio\.system\datasets\synapse_final.jsonl'

# Read raw bytes to check encoding
with open(path, 'rb') as f:
    raw = f.read(100)
    
# Check BOM
has_bom = raw[:3] == b'\xef\xbb\xbf'
print(f'Has UTF-8 BOM: {has_bom}')
print(f'First 50 bytes hex: {raw[:50].hex()}')

# Try reading as UTF-8
try:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f'UTF-8 read: OK, {len(lines)} lines')
except:
    print('UTF-8 read: FAILED')

# Check for broken chars in content
import json
broken = 0
ok = 0
for i, line in enumerate(lines):
    line = line.strip()
    if not line: continue
    try:
        d = json.loads(line)
        for m in d['messages']:
            c = m['content']
            # Check for replacement chars or black squares
            if '\ufffd' in c or '\u25a0' in c or '�' in c:
                broken += 1
                print(f'  Line {i}: BROKEN chars in {m["role"]}: {c[:100]}')
                break
        else:
            ok += 1
    except json.JSONDecodeError as e:
        print(f'  Line {i}: JSON ERROR: {e}')

print(f'\nOK lines: {ok}')
print(f'Broken lines: {broken}')

# Sample some Russian content to verify
import json
with open(path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i > 10: break
        d = json.loads(line.strip())
        for m in d['messages']:
            if m['role'] == 'assistant' and any('\u0400' <= c <= '\u04ff' for c in m['content']):
                print(f'\nSample RU (line {i}): {m["content"][:150]}')
                break
