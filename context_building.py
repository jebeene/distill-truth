import pandas as pd
from PROMPTS import *

def get_similar_statements(statement:str, num_of_sim_statement:int = 3) -> list[str]:
    ...

def get_classification_context_chain(
    row:pd.Series,
    shot_prompting:int,
    is_include_explanation:bool=True,
    is_include_speaker:bool=False,
    is_include_party:bool=False
) -> list[str]:

    statement = row[2]
    speaker = row[4]
    party = row[7]

    context = [
        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT.format(CLASSIFICATION_OPTIONS=CLASSIFICATION_OPTIONS)},
    ]

    if shot_prompting:
        context.extend(
            get_similar_statements(
                statement=statement,
                num_of_sim_statement=shot_prompting
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
):
    explanation = row['pred_explanation']
    context = [
        {"role": "system", "content": PE_SYSTEM_PROMPT.format(CLASSIFICATION_OPTIONS=CLASSIFICATION_OPTIONS)},
        {"role": "user", "content": PE_PROMPT.format(EXPLANATION=explanation)}
    ]

    return context


if __name__ == "__main__":
    df = pd.read_csv('./datasets/train.tsv', sep="\t")
    
    context = df.head(1).apply(
        lambda row: get_classification_context_chain(row, shot_prompting=False),
        axis=1
    )
