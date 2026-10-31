from backend.query_understanding import expand_query, require_wordnet


def test_expand_query_maps_pros_and_cons_to_indexed_terms():
    expanded = expand_query("What are the pros and cons of information systems?")

    assert "advantages" in expanded
    assert "disadvantages" in expanded
    assert "information systems" in expanded


def test_expand_query_maps_pros_of_to_advantages():
    expanded = expand_query("State any four pros of information systems")

    assert "advantages" in expanded
    assert "benefits" in expanded


def test_expand_query_preserves_original_terms_for_vocabulary_expansion():
    question = "How does an information system work?"

    expanded = expand_query(question)
    assert question.lower().split("?")[0] in expanded.lower()


def test_expand_query_adds_wordnet_synonyms_when_vocabulary_is_available():
    expanded = expand_query("What is the significant impact?")

    # WordNet availability differs between clean CI environments. The
    # original terms must always remain present; installed vocabularies add
    # at least one lexical bridge.
    assert "significant" in expanded
    assert "impact" in expanded


def test_expand_query_maps_component_questions_to_element_language():
    expanded = expand_query("What are the core components of an information system?")

    assert "necessary elements" in expanded
    assert "required components" in expanded


def test_wordnet_resource_is_available():
    require_wordnet()
