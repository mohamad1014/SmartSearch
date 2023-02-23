"""
This is the main.py file which will contain the whole implementation
Not needed to complicate things with anothe file at this stage as the modules used should make it quiet easy
"""

import json

corpus_path = "../data/scidocs/corpus.jsonl"

def transform_data_to_haystack_content(input_dict):
    """
    This function takes one line as a dict from the corpus.jsonl and transforms it into the format requested by
    haystack's document stores.
    All the data will be added to the metadata key and a singular content key will be created and used
    :param input_dict: the input dictionary to be used
    :return: output dictionary that is in the haystack requested format
    """

    for x in ['_id', 'title', 'text']:
        input_dict['metadata'][x] = input_dict[x]
        input_dict.pop(x)

    input_dict['content'] = input_dict['metadata']['title'] + '. ' + input_dict['metadata']['text']

    return input_dict


data = [json.loads(line) for line in open(corpus_path, 'r')]
data = [transform_data_to_haystack_content(line) for line in data]

data_test = data[:1000]

# Setting up haystack and the pipelines

from haystack.document_stores import InMemoryDocumentStore
document_store = InMemoryDocumentStore()

# Now, let's write the docs to our DB.
document_store.write_documents(data_test)

### Retriever


# Single encoder for example via sentence transformers
from haystack.nodes import EmbeddingRetriever

retriever = EmbeddingRetriever(document_store=document_store,
                               embedding_model="sentence-transformers/roberta-base-nli-stsb-mean-tokens", # choose any model from huggingface's models
                               use_gpu=True,
                               model_format="farm", # "sentence-transformers" is another option depending on model
                               )


# Important:
# Now that after we have the DPR initialized, we need to call update_embeddings() to iterate over all
# previously indexed documents and update their embedding representation.
# While this can be a time consuming operation (depending on corpus size), it only needs to be done once.
# At query time, we only need to embed the query and compare it the existing doc embeddings which is very fast.
document_store.update_embeddings(retriever)

### Pipeline
# Select the default pipeline for document search or build your own custom one (e.g. combining multiple retrievers).
# See details here: https://haystack.deepset.ai/docs/latest/pipelinesmd
from haystack.pipelines import DocumentSearchPipeline
pipe = DocumentSearchPipeline(retriever=retriever)

## Voilà! Ask a question!
prediction = pipe.run(query="Who is the father of Arya Stark?")
print(prediction)
# Returns list of docs

