"""Retain GraspGen inputs and raw proposals for recovery evidence."""
from pathlib import Path
import json, hashlib, os, functools, time


def seal(p):
    p = Path(p)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    return {'path': str(p), 'sha256': h}

def install_raw_archive(detector_class, row_root):
    original = detector_class._query_server

    @functools.wraps(original)
    def query(self, point_cloud, *, top_k):
        import numpy as np
        root = Path(row_root) / 'graspgen_raw'
        root.mkdir(parents=True, exist_ok=True)
        ordinal = int(getattr(self, '_seven_raw_ordinal', 0))
        self._seven_raw_ordinal = ordinal + 1
        d = root / (str(os.getpid()) + '_' + str(id(self)) + '_' + str(ordinal))
        d.mkdir(exist_ok=False)

        def array(name, value):
            p = d / name
            with p.open('xb') as f:
                np.save(f, np.asarray(value), allow_pickle=False)
                f.flush()
                os.fsync(f.fileno())
            return seal(p)
        info = {'schema': 'carryon-graspgen-raw-archive/v1', 'point_cloud': array('POINT_CLOUD.npy', point_cloud), 'top_k': int(top_k), 'started_wall_ts': time.time(), 'additional_model_calls': 0}
        try:
            result = original(self, point_cloud, top_k=top_k)
            if result is not None:
                info['raw_grasps'] = array('RAW_GRASPS_4X4.npy', result[0])
                info['raw_confidences'] = array('RAW_CONFIDENCES.npy', result[1])
            else:
                info['result_none'] = True
            return result
        finally:
            info['finished_wall_ts'] = time.time()
            p = d / 'ARCHIVE_RECEIPT.json'
            with p.open('x') as f:
                json.dump(info, f, sort_keys=True, indent=2)
                f.write('\n')
            seal(p)
            self.last_debug['raw_archive'] = info
    detector_class._query_server = query
