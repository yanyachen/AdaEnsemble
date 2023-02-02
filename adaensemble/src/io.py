import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops
'''
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
import apache_beam as beam
'''
import collections
import csv
from tqdm import tqdm


def count_lines(file_name):
    with open(file_name, "rb") as file:
        count = sum(1 for _ in file)
    return count


def build_csv_vocabulary(
    filenames, columns,
    vocabulary_folder, threshold
):
    total_rows = sum([count_lines(each)-1 for each in filenames])
    pbar = tqdm(total=total_rows)
    cnt_dict = {col: collections.Counter() for col in columns}

    for each_file in filenames:
        with open(each_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                for col in cnt_dict:
                    cnt_dict[col][row[col]] += 1
                pbar.update(1)

    for col in cnt_dict:
        with open(vocabulary_folder + "/" + col + ".txt", mode="w") as file:
            for level, count in cnt_dict[col].items():
                if count > threshold and level != "":
                    file.write(level + "\n")

    return None


def build_csv_dataset(
    filenames,
    feature_name,
    label_name,
    schema_dict,
    compression_type='',
    buffer_size=None,
    field_delim=",",
    use_quote_delim=True,
    na_value="",
    shuffle=False,
    shuffle_buffer_size=8192,
    num_epochs=1,
    batch_size=1024,
    num_parallel_calls=None,
    prefetch_buffer_size=1,
    feature_engineering_fn=None,
    **kwagrs
):
    # Column Name
    with open(filenames[0]) as file:
        first_line = file.readline()
        colname_list = first_line.strip().split(field_delim)
    # Preparing arguments
    selected_column_set = set(feature_name + label_name)
    selected_index, selected_colname = zip(*[
        (i, colname)
        for i, colname in enumerate(colname_list)
        if colname in selected_column_set
    ])
    selected_col_schema = [
        schema_dict[col] for col in selected_colname
    ]
    # Reading Dataset
    dataset = tf.data.experimental.CsvDataset(
        filenames=filenames,
        record_defaults=selected_col_schema,
        compression_type=compression_type,
        buffer_size=buffer_size,
        header=True,
        field_delim=field_delim,
        use_quote_delim=use_quote_delim,
        na_value=na_value,
        select_cols=selected_index
    )
    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer_size,
            seed=0,
            reshuffle_each_iteration=True
        )
    # Repeat
    if num_epochs > 1:
        dataset = dataset.repeat(num_epochs)

    # Batch
    dataset = dataset.batch(batch_size)

    # Mapping Function
    def map_fn(*columns):
        columns_dict = collections.OrderedDict(
            zip(selected_colname, columns)
        )
        features = {
            each: columns_dict[each]
            for each in feature_name
        }
        if len(label_name) > 1:
            labels = {
                each: columns_dict[each]
                for each in label_name
            }
        else:
            labels = columns_dict[label_name[0]]
        if feature_engineering_fn:
            features_new = feature_engineering_fn(features, **kwagrs)
            features.update(features_new)
        return features, labels

    # Mapping
    dataset = dataset.map(map_fn, num_parallel_calls)

    # Prefetch
    dataset = dataset.prefetch(prefetch_buffer_size)

    return dataset


def build_csv_dataset_parallel(
    filenames,
    feature_name,
    label_name,
    schema_dict,
    compression_type=None,
    buffer_size=None,
    field_delim=",",
    use_quote_delim=True,
    na_value="",
    num_parallel_reads=1,
    sloppy=False,
    shuffle=False,
    shuffle_buffer_size=8192,
    num_epochs=1,
    batch_size=1024,
    num_parallel_calls=None,
    prefetch_buffer_size=1,
    feature_engineering_fn=None,
    **kwagrs
):
    # Column Name
    with open(filenames[0]) as file:
        first_line = file.readline()
        colname_list = first_line.strip().split(field_delim)

    # Preparing arguments
    selected_column_set = set(feature_name + label_name)
    selected_index, selected_colname = zip(*[
        (i, colname)
        for i, colname in enumerate(colname_list)
        if colname in selected_column_set
    ])
    selected_col_schema = [
        schema_dict[col] for col in selected_colname
    ]

    # Single Reading Dataset Function
    def filename_to_dataset(filename):
        return tf.contrib.data.CsvDataset(
            filenames=filename,
            record_defaults=selected_col_schema,
            compression_type=compression_type,
            buffer_size=buffer_size,
            header=True,
            field_delim=field_delim,
            use_quote_delim=use_quote_delim,
            na_value=na_value,
            select_cols=selected_index
        )

    # Mapping Function
    def map_fn(*columns):
        columns_dict = collections.OrderedDict(
            zip(selected_colname, columns)
        )
        features = {
            each: columns_dict[each]
            for each in feature_name
        }
        labels = {
            each: columns_dict[each]
            for each in label_name
        }
        if feature_engineering_fn:
            features_new = feature_engineering_fn(features, **kwagrs)
            features.update(features_new)
        return features, labels

    # Parallel Reading
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            filename_to_dataset,
            cycle_length=num_parallel_reads,
            sloppy=sloppy
        )
    )
    dataset = _maybe_shuffle_and_repeat(
        dataset,
        num_epochs=num_epochs,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        shuffle_seed=0
    )
    dataset = dataset.batch(
        batch_size=batch_size
    )
    dataset = dataset_ops.ParallelMapDataset(
        dataset,
        map_func=map_fn,
        num_parallel_calls=num_parallel_calls
    )
    dataset = dataset.prefetch(prefetch_buffer_size)

    return dataset


def _maybe_shuffle_and_repeat(
    dataset,
    num_epochs,
    shuffle,
    shuffle_buffer_size,
    shuffle_seed
):
    if num_epochs != 1 and shuffle:
        return dataset.apply(
            tf.data.experimental.shuffle_and_repeat(
                shuffle_buffer_size, num_epochs, shuffle_seed
            )
        )
    elif shuffle:
        return dataset.shuffle(shuffle_buffer_size, shuffle_seed)
    elif num_epochs != 1:
        return dataset.repeat(num_epochs)
    return dataset


def tft_csv_to_tfrecord(
    input_pattern,
    output_prefix,
    temp_dir,
    batch_size,
    input_metadata,
    mode,
    transform_func_path,
    preprocessing_fn,
    output_metadata_path,
    tfrecord_num_shards,
):
    # CSV Header
    with open(tf.gfile.Glob(input_pattern)[0]) as f:
        csv_header = f.readline().strip().split(',')

    # Pipeline
    with beam.Pipeline() as pipeline:
        with tft_beam.Context(
            temp_dir=temp_dir,
            desired_batch_size=batch_size
        ):

            # Input Coder
            csv_coder = tft.coders.CsvCoder(
                csv_header, input_metadata.schema
            )

            # Input Data
            raw_data = (
                pipeline
                | 'ReadFromText' >> beam.io.ReadFromText(
                    input_pattern, skip_header_lines=1
                )
                | 'ParseCSV' >> beam.Map(csv_coder.decode)
            )

            # TransformFn
            if mode == 'train':
                transform_fn = (
                    (raw_data, input_metadata)
                    | 'Analyze' >> tft_beam.AnalyzeDataset(preprocessing_fn)
                )

                _ = (
                    transform_fn
                    | 'WriteTransformFn' >> tft_beam.WriteTransformFn(
                        transform_func_path
                    )
                )

            elif mode == 'test':
                transform_fn = pipeline | tft_beam.ReadTransformFn(
                    transform_func_path
                )

            # TransformDataset
            (transformed_data, transformed_metadata) = (
                ((raw_data, input_metadata), transform_fn)
                | 'Transform' >> tft_beam.TransformDataset()
            )

            # DatasetMetadata
            if mode == 'train' and output_metadata_path is not None:
                _ = (
                    transformed_metadata
                    | 'WriteMetadata' >> tft_beam.WriteMetadata(
                        output_metadata_path, pipeline
                    )
                )

            # Output Coder
            tfrecord_coder = tft.coders.ExampleProtoCoder(
                transformed_metadata.schema
            )

            # WriteToTFRecord
            _ = (
                transformed_data
                | 'SerializeExamples' >> beam.Map(tfrecord_coder.encode)
                | 'WriteExamples' >> beam.io.WriteToTFRecord(
                    output_prefix,
                    num_shards=tfrecord_num_shards
                )
            )

    return None


def build_tfrecord_dataset(
    filenames,
    input_metadata,
    feature_name,
    label_name,
    buffer_size,
    num_parallel_reads,
    batch_size,
    num_epochs,
    shuffle,
    shuffle_buffer_size,
    num_parallel_calls,
    prefetch_buffer_size
):
    # Parser
    feature_spec_dict = input_metadata.schema.as_feature_spec()
    feature_spec_dict = {
        key: feature_spec_dict[key]
        for key in feature_name + label_name
    }

    def parser(example):
        parsed_example = tf.parse_single_example(example, feature_spec_dict)
        features = {each: parsed_example[each] for each in feature_name}
        if len(label_name) == 1:
            labels = parsed_example[label_name[0]]
        else:
            labels = {each: parsed_example[each] for each in label_name}
        return features, labels

    # Dataset
    dataset = tf.data.TFRecordDataset(
        filenames=filenames,
        buffer_size=buffer_size,
        num_parallel_reads=num_parallel_reads
    )

    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer_size,
            seed=0,
            reshuffle_each_iteration=True
        )
    # Repeat
    if num_epochs > 1:
        dataset = dataset.repeat(num_epochs)

    # Mapping
    dataset = dataset.map(parser, num_parallel_calls)

    # Batch
    dataset = dataset.batch(batch_size)

    # Prefetch
    dataset = dataset.prefetch(prefetch_buffer_size)

    return dataset
