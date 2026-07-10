from statebudgetmem.baselines.memorybank.datasets import (
    load_reproduction_dataset
)


def test_memorybank_reproduction_dataset():

    users, probes = load_reproduction_dataset(
        "data/memorybank_reproduction"
    )

    assert len(users) == 5
    assert len(probes) == 50


    user_ids = {
        user.user_id
        for user in users
    }

    assert user_ids == {
        "user_001",
        "user_002",
        "user_003",
        "user_004",
        "user_005"
    }