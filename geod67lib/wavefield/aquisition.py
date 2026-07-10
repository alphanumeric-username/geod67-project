import json

import tensorflow as tf

import numpy as np

class AquisitionParameters:
    def __init__(self, src_positions, rec_positions, dt, tn, src_wavelet):
        self.src_positions = src_positions
        self.rec_positions = rec_positions
        self.dt = dt
        self.tn = tn//dt * dt

        self.src_wavelet = src_wavelet
        self.rec_data = tf.zeros((self.rec_positions.shape[0], 0), dtype=src_wavelet.dtype)
        
        self._last_rec_pos = None
        self._last_src_pos = None
        self._rec_indices_cache = None
        self._src_indices_cache = None


    @property
    def nt(self):
        return int(self.tn//self.dt)

    @property
    def nrec(self):
        return self.rec_positions.shape[0]

    @property
    def nsrc(self):
        return self.src_positions.shape[0]

    def src(self, ti):
        if len(self.src_positions.shape) == 2:
            if ti < 0 or ti >= len(self.src_wavelet[0]):
                return 0
            return self.src_wavelet[:, ti]
        elif ti < 0 or ti >= len(self.src_wavelet):
            return 0
        return self.src_wavelet[ti]
    
    @property
    def rec_indices(self):
        if (self._last_rec_pos is None) or not(bool(np.prod(self._last_rec_pos == self.rec_positions))):
            indices = []
            for r in range(self.rec_positions.shape[0]):
                xr = np.array(self.rec_positions[r], dtype=np.int32).tolist()
                xr.insert(0, 0)
                xr.append(0)
                xr = tuple(xr)
                indices.append(xr)
            self._rec_indices_cache = indices
            self._last_rec_pos = self.rec_positions
        return self._rec_indices_cache

    @property
    def src_indices(self):
        pos = []
        for s in range(self.src_positions.shape[0]):
            xs = self.src_positions[s]
            xs = (0, xs[0], xs[1], 0)
            pos.append(xs)
            
        return pos

    def update_rec_data(self, ti, u):
        recs_at_ti = tf.gather_nd(u, indices=self.rec_indices)
        recs_at_ti = tf.reshape(recs_at_ti, (recs_at_ti.shape[0], 1))
        self.rec_data = tf.concat([self.rec_data, recs_at_ti], axis=-1)
    
    
    def reset(self):
        self.rec_data = np.zeros((self.rec_positions.shape[0], 0), dtype=self.src_wavelet.dtype)
    

    def adjoint_parameters(self):
        """
        Swaps
        """
        return AquisitionParameters(1*self.rec_positions, 1*self.src_positions, self.dt, self.tn, np.flip(self.rec_data.numpy(), axis=1))


    def copy(self):
        return AquisitionParameters(1*self.src_positions, 1*self.rec_positions, self.dt, self.tn, 1*self.src_wavelet)

    
    def save(self, path):
        with open(path, 'w+') as fout:
            ap_data = {
                'src_positions': self.src_positions.tolist(),
                'rec_positions': self.rec_positions.tolist(),
                'dt': self.dt,
                'tn': self.tn,
                'src_wavelet': self.src_wavelet.tolist()
            }

            json.dump(ap_data, fout, indent=4)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as fin:
            ap_data = json.load(fin)
            ap = AquisitionParameters(
                np.array(ap_data['src_positions'], np.float32),
                np.array(ap_data['rec_positions'], np.float32),
                ap_data['dt'],
                ap_data['tn'],
                np.array(ap_data['src_wavelet'], np.float32)
            )
            # ap.rec_data = np.array(ap_data['rec_data'], dtype=np.float32)
            return ap