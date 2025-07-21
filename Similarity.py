from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


def get_embeddings(russian_paragraphs, persian_paragraphs):
    russian_embeddings = model.encode(russian_paragraphs, convert_to_tensor=True)
    persian_embeddings = model.encode(persian_paragraphs, convert_to_tensor=True)
    return russian_embeddings, persian_embeddings


def get_similarity(russian_embeddings, persian_embeddings):
    if russian_embeddings.device.type != 'cuda':
        russian_embeddings = russian_embeddings.to("cuda")
    if persian_embeddings.device.type != 'cuda':
        persian_embeddings = persian_embeddings.to("cuda")
    print(russian_embeddings.device.type, persian_embeddings.device.type)
    return util.pytorch_cos_sim(russian_embeddings, persian_embeddings)