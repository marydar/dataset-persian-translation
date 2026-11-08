
import time
import random


def random_delay(
    low,
    high
):

    time.sleep(
        random.uniform(
            low,
            high
        )
    )
