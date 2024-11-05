import random
import objaverse
import multiprocessing as mp

random.seed(32)
uids = objaverse.load_uids()

random_object_uids = random.sample(uids, 100000)


processes = mp.cpu_count()
objects = objaverse.load_objects(
    uids=random_object_uids,
    download_processes=processes)