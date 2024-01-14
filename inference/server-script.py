import json

def process_ner_output(ner_output):
    # Process NER output and extract author and date with the highest score
    author = None
    date = None

    for entity in ner_output:
        if entity['entity_group'] == 'AUTHOR' and (author is None or entity['score'] > author['score']):
            author = {'author': entity['word'], 'score': entity['score']}

        if entity['entity_group'] == 'DATE' and (date is None or entity['score'] > date['score']):
            date = {'date': entity['word'], 'score': entity['score']}

    return {'author': author['author'] if author else 'N/A', 'date': date['date'] if date else 'N/A'}


# list: simulates the json that we receive from the frontend. In this case, it is highly recommended to use the examples from the test-dataset. 
# The model is still not able to handle all the possible inputs, since the training dataset has not been optimized yet. Test dataset can be found in the folder: test.jsonl.
def process_text_list(text_list):
    result_list = []
    ner_output_list = []
    for i in range(len(text_list)):
        ner_output_list.append(classifier(text_list[i]))

    for text, ner_output in zip(text_list, ner_output_list):
        processed_result = process_ner_output(ner_output)
        result_list.append({'text': text, 'response': processed_result})

    return json.dumps(result_list, indent=2)






if __name__ == '__main__':
    # PREREQUESITES:

    model_id = 'textminr/ner-multilingual-bert'
    from transformers import pipeline
    classifier = pipeline(
      'ner',
      model=model_id,
      aggregation_strategy='simple'
    )

    # EXAMPLE
    texts = ["Back in 2008, when uncertainty clouded my path, I discovered solace within the pages of 'Beyond the Horizon' by Alan Walker. As Oprah Winfrey remarked in 2010, it was 'transformative'.",
            "In einer Studie aus dem Jahr 1985 wurde die Wirkung von Umweltverschmutzung auf Oekosysteme untersucht.",
            "The podcast 'History Uncovered' released an episode on January 27, 2022, in which host James Fisher interviewed renowned historian Dr. Karen Miller about the fall of the Roman Empire (Fisher & Miller, 2022)."]


    model_id = 'textminr/ner-multilingual-bert'
    from transformers import pipeline
    classifier = pipeline(
    'ner',
    model=model_id,
    aggregation_strategy='simple'
    )

    result = process_text_list(texts)
    print(result)
