import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'c:\projects\VKVstudio\training\stress_test_dialogues.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

stack_terms = {
    'svelte': 0, 'astro': 0, 'gsap': 0, 'lenis': 0, 
    'lighthouse': 0, 'typescript': 0, 'webgpu': 0, 'onnx': 0,
    'qlora': 0, 'ollama': 0, 'gemma': 0
}

total = 0
stack_in_offtopic = []

for d in data:
    for t in d['turns']:
        total += 1
        a = t['assistant'].lower()
        for term in stack_terms:
            if term in a:
                stack_terms[term] += 1
        
        if d['persona'] in ('troll', 'grandma', 'sad', 'child', 'spammer', 'philosopher'):
            mentioned = [term for term in ['svelte', 'astro', 'gsap', 'lighthouse'] if term in a]
            if mentioned and 'svelte' not in t['user'].lower() and 'astro' not in t['user'].lower():
                stack_in_offtopic.append({
                    'id': d['id'], 'persona': d['persona'],
                    'turn': t['turn'],
                    'user': t['user'][:100],
                    'mentioned': mentioned,
                    'snippet': t['assistant'][:200]
                })

print('STACK TERM FREQUENCY IN ALL RESPONSES:')
print('Total turns:', total)
for term, count in sorted(stack_terms.items(), key=lambda x: -x[1]):
    pct = 100*count/total
    print(f'  {term:15s}: {count:4d} ({pct:5.1f}%)')

print(f'\nSTACK IN OFF-TOPIC CONTEXTS ({len(stack_in_offtopic)} cases):')
for ex in stack_in_offtopic[:20]:
    did = ex['id']
    per = ex['persona']
    trn = ex['turn']
    ment = ex['mentioned']
    usr = ex['user']
    snip = ex['snippet']
    print(f'  [D{did}] {per} turn {trn}: {ment}')
    print(f'    USER: {usr}')
    print(f'    SYNAPSE: {snip}...')
    print()

# Force-redirect count
force_stack = 0
total_offtopic = 0
for d in data:
    if d['persona'] in ('troll','grandma','sad','child','spammer'):
        for t in d['turns']:
            total_offtopic += 1
            a = t['assistant'].lower()
            u = t['user'].lower()
            if ('astro' in a or 'svelte' in a) and 'astro' not in u and 'svelte' not in u:
                force_stack += 1

pct = 100*force_stack//max(total_offtopic,1)
print(f'OFF-TOPIC force-redirect to stack: {force_stack}/{total_offtopic} ({pct}%)')

# Check bland/boring responses more carefully
print('\n\nBLAND RESPONSE ANALYSIS:')
bland_count = 0
total_long = 0
for d in data:
    for t in d['turns']:
        a = t['assistant']
        if len(a) < 200:
            continue
        total_long += 1
        # Check for life/personality
        life_markers = [
            '😊', '👋', '🔥', '💡', '⚡', '🎯', '🏗', '!',
            'хаха', 'haha', 'кстати', 'знаешь', 'знаешь что',
            'а вот', 'смотри', 'прикинь', 'класс', 'круто',
            'you know', 'fun fact', 'cool', 'nice',
            'представь', 'imagine', 'think of it',
        ]
        has_life = sum(1 for m in life_markers if m in a.lower())
        if has_life < 2:
            bland_count += 1

print(f'  Long responses (>200 chars): {total_long}')
print(f'  Bland (< 2 personality markers): {bland_count} ({100*bland_count//max(total_long,1)}%)')
