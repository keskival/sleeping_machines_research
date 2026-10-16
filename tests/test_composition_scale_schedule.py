from experiments.hierarchy.composition_scale_learning import PairSchedule


def test_pair_schedule_covers_every_pair_before_reuse_and_resumes_across_epoch():
 schedule=PairSchedule(1024);first=[schedule.next() for _ in range(997)];saved=schedule.state_dict()
 tail=[schedule.next() for _ in range(1051)];restored=PairSchedule(1024);restored.load_state_dict(saved)
 assert tail==[restored.next() for _ in range(1051)]
 sequence=first+tail
 assert len(set(sequence[:1024]))==len(set(sequence[1024:]))==1024
 assert schedule.epochs==restored.epochs==2
