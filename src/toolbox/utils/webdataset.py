from webdataset import filters
from webdataset.handlers import reraise_exception
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

trace = False

def group_by_keys(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Group key, value pairs into samples.
    
    Code borrowed from the webdataset group_by_keys() function, modified to handle
    duplicate file names.

    Args:
        data: Iterator over tarfile contents.
        keys: Function that takes a file name and returns a key and a suffix.
            Defaults to base_plus_ext.
        lcase: Whether to lowercase the suffix. Defaults to True.
        suffixes: List of suffixes to keep. Defaults to None.
        handler: Exception handler. Defaults to None.

    Raises:
        ValueError: Raised if there are duplicate file names in the tar file.

    Yields:
        Iterator over samples.
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if trace:
            print(
                prefix,
                suffix,
                current_sample.keys() if isinstance(current_sample, dict) else None,
            )
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        if current_sample is None or prefix != current_sample["__key__"]:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = {"__key__": prefix, "__url__": filesample["__url__"]}
            
        if suffix in current_sample:
            # Handle duplicate file names in different shards:
            # instead of raising an exception, the sample is marked as
            # bad and the duplicate file is ignored
            print(
                f"{fname}: duplicate file name in tar file {suffix} "
                f"{current_sample.keys()}",
            )
            current_sample["__bad__"] = True
        
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    
    # Discard duplicates
    if valid_sample(current_sample):
        yield current_sample


def tarfile_samples(src, handler=reraise_exception):
    """Given a stream of tar files, yield samples.
    
    Code borrowed from the webdataset tarfile_samples() function.

    Args:
        src (_type_): Stream of tar files.
        handler (_type_, optional): Exception handler. Defaults to reraise_exception.

    Returns:
        _type_: Stream of samples.
    """
    streams = url_opener(src, handler=handler)
    
    # NOTE: I could use the select_files parameter to split the samples into train,
    # validation, and test sets. SAMPLE LEVEL SPLITTING.
    files = tar_file_expander(streams, handler=handler, select_files=None)
    samples = group_by_keys(files, handler=handler)
    
    return samples

tarfile_to_samples = filters.pipelinefilter(tarfile_samples)
