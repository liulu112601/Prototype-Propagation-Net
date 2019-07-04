import os, sys
from pathlib import Path
import numpy as np
import scipy.misc
import pprint

pp = pprint.PrettyPrinter(indent=4)

try:
  from StringIO import StringIO  # Python 2.7
except ImportError:
  from io import BytesIO  # Python 3.x

class Logger(object):
  def __init__(self, log_dir, seed, use_tf=False):
    """Create a summary writer logging to log_dir."""
    self.log_dir = Path("{:}/{:}".format(str(log_dir),seed))
    if not self.log_dir.exists(): os.makedirs(str(self.log_dir))

    self.baseline_classifier_dir = self.log_dir / 'baseline_classifier'
    if not self.baseline_classifier_dir.exists(): os.makedirs(str(self.baseline_classifier_dir))

    self.log_file = '{:}/log-{:}.txt'.format(self.log_dir, seed)
    self.file_writer = open(self.log_file, 'w')
    
    if use_tf:
      self.writer = tf.summary.FileWriter(str(self.event_dir))

  def print(self, string, fprint=True, is_pp=False):
    if is_pp: pp.pprint (string)
    else:     print(string)
    if fprint:
      self.file_writer.write('{:}\n'.format(string))
      self.file_writer.flush()

  def close(self):
    self.file_writer.close()

  def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""
    import tensorflow as tf
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    self.writer.add_summary(summary, step)

  def histo_summary(self, tag, values, step, bins=1000):
    """Log a histogram of the tensor of values."""
    import tensorflow as tf

    # Create a histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill the fields of the histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values ** 2))

    # Drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
      hist.bucket_limit.append(edge)
    for c in counts:
      hist.bucket.append(c)

    # Create and write Summary
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    self.writer.add_summary(summary, step)
    self.writer.flush()

  def log(logger, names, values, step):
    import tensorflow as tf

    if not isinstance(names, list):
      names = [names]
    if not isinstance(values, list):
      values = [values]
    assert len(names) == len(values), '{:} vs {:}'.format(len(names), len(values))
    info = [ (name, value) for name, value in zip(names, values)]
    info = dict(info) 

    for tag, value in info.items():
      logger.scalar_summary(tag, value, step + 1)
