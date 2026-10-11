"""Joint content/time packet retaining native parameter graph for actor training."""
from native_temporal_teaching_packet import NativeTemporalTeachingPacket
from native_training_linearization import NativeTrainingLinearization
class NativeTrainingTemporalPacket(NativeTemporalTeachingPacket,NativeTrainingLinearization):
    pass
