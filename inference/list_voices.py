import edge_tts
import asyncio

async def main():
    voices = await edge_tts.list_voices()
    for v in voices:
        if 'Multilingual' in v['ShortName'] or 'ru-RU' in v['Locale'] or 'en-US' in v['Locale']:
            print(v['ShortName'])

asyncio.run(main())
