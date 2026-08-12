from pathlib import Path

data_dir = Path('data')

documents = []

for file_path in data_dir.glob('*.txt'):
    text = file_path.read_text(encoding='utf-8')

    documents.append({
        'text' : text,
        'metadata' : {
            'source' : file_path.name
        }
    })

for document in documents:
    print('=' * 50)
    print('Source:', document['metadata']['source'])
    print(document['text'])