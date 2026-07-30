from data_processing.text_augmentation import TextAugmenter, weak_view


def tags(tokens):
    return ["RB" if token.lower() == "never" else "NN" for token in tokens]


def synonyms(token, _tag):
    return [] if token.lower() == "never" else [f"{token}_syn"]


def augmenter():
    return TextAugmenter(
        replacement_rate=0.10,
        insertion_rate=0.05,
        edit_cap=0.20,
        synonym_provider=synonyms,
        pos_tagger=tags,
        stopwords=frozenset({"the"}),
    )


def test_determinism_boundaries_and_original_unchanged():
    first = "cats never chase small birds quickly"
    second = "dogs watch the quiet garden"
    original = (first, second)
    one = augmenter().augment_pair(first, second, 7, 3, "train:00001")
    two = augmenter().augment_pair(first, second, 7, 3, "train:00001")
    assert one == two
    assert original == (first, second)
    assert "never" in one[0]
    assert one[0] and one[1]
    assert not any(token.startswith("dogs") for token in one[0].split())
    assert not any(token.startswith("cats") for token in one[1].split())


def test_epoch_can_change_stream_and_weak_is_identity():
    sentence = "alpha beta gamma delta epsilon zeta eta theta"
    outputs = {augmenter().augment_sentence(sentence, 1, epoch, "x", 1) for epoch in range(8)}
    assert len(outputs) > 1
    assert weak_view("a", "b") == ("a", "b")


def test_edit_budget():
    sentence = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    output = augmenter().augment_sentence(sentence, 1, 2, "x", 1)
    changed = sum("_syn" in token for token in output.split())
    assert changed <= 2
