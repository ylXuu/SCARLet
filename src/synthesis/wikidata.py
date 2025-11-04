''' search related entities on wikidata '''

from typing import List
import spacy
import requests
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_entities(text: str):
    ''' parse entities from text '''

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    entities = [entity.text for entity in doc.ents]

    return entities


def fetch_wikidata_api_wbsearchentities(search: str,
                                        language: str = 'en',
                                        limit: int = 1,
                                        return_raw: bool = False,
                                        return_title_desc: bool = False):
    '''
    fuzzy search entities in wikidata by text.
    - search: text
    - limit: maximum return number
    - return_raw: if true, return raw json, else return entity's id
    - return_title_desc: if true, return title and description of the entity
    '''

    url = 'https://www.wikidata.org/w/api.php'
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'search': search,
        'language': language,
        'type': 'item',    
        'limit': limit
    }

    try:
        response = requests.get(url, params=params).json()
        if return_raw:
            return response
        else:
            if response.get('search', {}) == []:
                return None
            else:
                if return_title_desc:
                    return (
                        response['search'][0]['id'],
                        response['search'][0]['display']['label']['value'],
                        response['search'][0]['display']['description']['value']
                    )
                else:
                    return response['search'][0]['id']
    except Exception as e:
        logger.error(f'There was an error: {e}')
        return None


def fetch_wikidata_api_wbgetentities(id: str,
                                     language: str = 'en',
                                     return_raw = False,):
    '''
    get entity's information in wikidata by id.
    - return_raw: if true, return raw json, else return processed json
    '''

    url = 'https://www.wikidata.org/w/api.php'
    params = {
        'action': 'wbgetentities',
        'ids': id,
        'format': 'json',
        'language': language,
    }

    try:
        response = requests.get(url, params=params).json()
        if return_raw:
            return response
        if 'error' in response:
            # search fail - no matching entity
            return None
        else:
            data = {}
            # entity's name
            try:
                data['title'] = response['entities'][id]['labels'][language]['value']
            except:
                data['title'] = None
            # entity's decriptions
            try:
                data['descriptions'] = response['entities'][id]['descriptions'][language]['value']
            except:
                data['descriptions'] = None
            # entity's instance_of
            try:
                data['instance_of'] = [v['mainsnak']['datavalue']['value']['id'] for v in response['entities'][id]['claims']['P31']]
            except:
                data['instance_of'] = None
            # entity's part_of
            try:
                data['part_of'] = [v['mainsnak']['datavalue']['value']['id'] for v in response['entities'][id]['claims']['P361']]
            except:
                data['part_of'] = None
            # entity's founded_by
            try:
                data['founded_by'] = [v['mainsnak']['datavalue']['value']['id'] for v in response['entities'][id]['claims']['P112']]
            except:
                data['founded_by'] = None
            # entity's categories
            try:
                data['categories'] = [v['mainsnak']['datavalue']['value']['id'] for v in response['entities'][id]['claims']['P910']]
            except:
                data['categories'] = None
            # entity's inception
            try:
                data['inception'] = response['entities'][id]['claims']['P571'][0]['mainsnak']['datavalue']['value']['time']
            except:
                data['inception'] = None
            # entity's has_part  P527
            try:
                # print(json.dumps(response['entities'][id]['claims']['P527'], indent=4)) # for recognizing json structure
                data['has_part'] = [v['mainsnak']['datavalue']['value']['id'] for v in response['entities'][id]['claims']['P527']]
            except:
                data['has_part'] = None
            # entity's has_part_of_the_class  P2670
            try:
                data['has_part_of_the_class'] = [v['mainsnak']['datavalue']['value']['id'] for v in response['entities'][id]['claims']['P2670']]
            except:
                data['has_part_of_the_class'] = None
            
            return data
    except Exception as e:
        logger.error(f'There was an error: {e}')
        return None


def fetch_wikidata_all_sparql(id: str,
                              language: str = 'en',
                              limit: int = 10):
    '''
    get entity's information in wikidata by id via SPARQL.
    '''

    sparql_query = '''
    SELECT ?property ?propertyLabel ?object ?objectLabel
    WHERE {
      wd:{id} ?property ?object.
      ?property rdfs:label ?propertyLabel.
      ?object rdfs:label ?objectLabel.
      FILTER(LANG(?propertyLabel) = "{language}")
      FILTER(LANG(?objectLabel) = "{language}")
    }
    LIMIT {limit}
    '''.format(id=id, language=language, limit=limit)
    endpoint_url = 'https://query.wikidata.org/sparql'

    try:
        response = requests.get(endpoint_url, params={'query': sparql_query, 'format': 'json'}).json()
        return response
    except Exception as e:
        logger.error(f'There was an error: {e}')
        return None


def fetch_wikidata_instance_of_sparql(id: str,
                                      language: str = 'en',
                                      limit: int = 10):
    '''
    get entity's information of 'instance_of' in wikidata by id via SPARQL.
    '''

    sparql_query = '''
    SELECT ?item ?itemLabel
    WHERE {
      ?item wdt:P31 wd:{id};
            rdfs:label ?itemLabel.
      FILTER(LANG(?itemLabel) = "{language}")
    }
    LIMIT {limit}
    '''.format(id=id, language=language, limit=limit)
    endpoint_url = 'https://query.wikidata.org/sparql'

    try:
        response = requests.get(endpoint_url, params={'query': sparql_query, 'format': 'json'}).json()
        return response
    except Exception as e:
        logger.error(f'There was an error: {e}')
        return None


def collect_entities(seed_entities: List,
                     max_related: int = 5):
    '''
    collect related entities for subsequent context construction.
    - max_related: the maximum number of related entities considered for each entity
    '''

    related_entities_list = []
    for entity in seed_entities:
        related_entities = []

        entity_info = fetch_wikidata_api_wbsearchentities(search=entity, return_title_desc=True)
        if entity_info is None:
            # no corresponding entity, only save itself
            related_entities_list.extend([entity])
            continue
        entity_id = entity_info[0]
        related_entities.append(entity_info[1] + ': ' + entity_info[2])

        response = fetch_wikidata_api_wbgetentities(id=entity_id)
        if response is None:
            related_entities_list.extend(related_entities)
            continue

        related_entities_ids = []
        for key in list(set(response.keys()) - set(['title', 'descriptions'])):
            if response[key] is not None:
                for raw_id in response[key]:
                    related_entities_ids.append(raw_id)
        related_entities_ids = related_entities_ids[:max_related]

        for related_entity_id in related_entities_ids:
            response = fetch_wikidata_api_wbgetentities(id=related_entity_id)
            if response is None or response['title'] is None:
                continue
            if response['descriptions'] is None:
                title_desc = response['title']
            else:
                title_desc = response['title'] + ': ' + response['descriptions']
            related_entities.append(title_desc)
        
        related_entities_list.extend(related_entities)
    
    return related_entities_list


if __name__ == '__main__':
    # print(parse_entities(text='2006'))
    seed_entities = ['Bill Cosby']
    collect_entities(seed_entities)

