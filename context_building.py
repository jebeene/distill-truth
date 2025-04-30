import pandas as pd
from PROMPTS import *
from sentence_transformers import SentenceTransformer, util

def get_similar_statements(
    statement:str,
    top_k:int=3,
    is_include_speaker:bool=False,
    is_include_party:bool=False,
    is_include_explanation:bool=False,
) -> list[str]:
    # Load LIAR TSV dataset
    df_train = pd.read_csv("./datasets/train.tsv", sep="\t")
    # Use only necessary columns
    df_train = df_train[["statement", "label"]].dropna()
    # Load sentence transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_embeddings = model.encode(df_train['statement'].tolist(), convert_to_tensor=True)
    query_embedding = model.encode(statement, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    top_k_indices = [hit['corpus_id'] for hit in hits[0]]
    top_examples = df_train.iloc[top_k_indices]

    # Format as chat messages
    return [
        {
            "role": "user",
            "content": STATEMENT_CLASSIFICATION_PROMPT.format(
                STATEMENT=row['statement'],
                IS_INCLUDE_SPEAKER=IS_INCLUDE_SPEAKER.format(SPEAKER=row['speaker']) if is_include_speaker else '',
                IS_INCLUDE_PARTY=IS_INCLUDE_PARTY.format(PARTY_AFFILIATION=row['party']) if is_include_party else '',
                IS_INCLUDE_EXPLANATION=IS_INCLUDE_EXPLANATION if is_include_explanation else '',
                CLASSIFICATION_OPTIONS=CLASSIFICATION_OPTIONS
            )
        }
        for _, row in top_examples.iterrows()
    ]
    

def get_classification_context_chain(
    row:pd.Series,
    shot_prompting:int,
    is_include_explanation:bool=True,
    is_include_speaker:bool=False,
    is_include_party:bool=False
) -> list[str]:

    statement = row['statement']
    speaker = row['speaker']
    party = row['party']

    context = [
        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT.format(CLASSIFICATION_OPTIONS=CLASSIFICATION_OPTIONS)},
    ]

    if shot_prompting:
        context.extend(
            get_similar_statements(
                statement=statement,
                top_k=shot_prompting,
                is_include_speaker=is_include_speaker,
                is_include_party=is_include_party,
                is_include_explanation=is_include_explanation,
            )
        )

    context.append(
        {"role": "user", "content": STATEMENT_CLASSIFICATION_PROMPT.format(
            STATEMENT=statement,
            IS_INCLUDE_SPEAKER=IS_INCLUDE_SPEAKER.format(SPEAKER=speaker) if is_include_speaker else '',
            IS_INCLUDE_PARTY=IS_INCLUDE_PARTY.format(PARTY_AFFILIATION=party) if is_include_party else '',
            IS_INCLUDE_EXPLANATION=IS_INCLUDE_EXPLANATION if is_include_explanation else '',
            CLASSIFICATION_OPTIONS=CLASSIFICATION_OPTIONS
        )}
    )

    return context

def get_PE_context_chain(
    row: pd.Series
) -> list[str]:
    explanation = row['pred_explanation']
    context = [
        {"role": "system", "content": PE_SYSTEM_PROMPT.format(CLASSIFICATION_OPTIONS=CLASSIFICATION_OPTIONS)},
        {"role": "user", "content": PE_PROMPT.format(EXPLANATION=explanation)}
    ]

    return context


if __name__ == "__main__":
    df = pd.read_csv('./datasets/train.tsv', sep="\t")
    
    context = df.head(1).apply(
        lambda row: get_classification_context_chain(row, shot_prompting=2),
        axis=1
    )
