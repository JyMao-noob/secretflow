import logging
import os
from typing import List

from secretflow.component.component import (
    CompEvalError,
    Component,
    IoType,
    TableColParam,
)
from secretflow.component.data_utils import (
    DistDataType,
    download_files,
    extract_distdata_info,
    merge_individuals_to_vtable,
)
from secretflow.device.device.spu import SPU
from secretflow.spec.v1.data_pb2 import DistData, IndividualTable

# 声明组件
pir_comp = Component(
    "pir",
    domain="data_prep",
    version="0.0.1",
    desc="PIR between two parties.",
)

pir_comp.int_attr(
    name="num_per_query",
    desc="The number of queries processed by the server each time",
    is_list=False,
    is_optional=True,
    default_value=2,
    lower_bound=1,
    lower_bound_inclusive=True,
    upper_bound=None,
)

pir_comp.int_attr(
    name="label_max_len",
    desc="The limition of label length (byte)",
    is_list=False,
    is_optional=True,
    default_value=200,
    lower_bound=20,
    lower_bound_inclusive=True,
    upper_bound=None,
)

pir_comp.int_attr(
    name="bucket_size",
    desc="Indistinguishable degree",
    is_list=False,
    is_optional=True,
    default_value=1000000,
    lower_bound=1000000,
    lower_bound_inclusive=True,
    upper_bound=None,
)

pir_comp.io(
    io_type=IoType.INPUT,
    name="client_query_data_input",
    desc="Individual table for query",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="key",
            desc="Column used to query.",
            col_min_cnt_inclusive=1,
        )
    ],
)

pir_comp.io(
    io_type=IoType.INPUT,
    name="server_data_input",
    desc="Individual table as server input data",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="label",
            desc="Columns used as label.",
            col_min_cnt_inclusive=1,
        )
    ],
)

pir_comp.io(
    io_type=IoType.OUTPUT,
    name="pir_output",
    desc="Output vertical table",
    types=[DistDataType.VERTICAL_TABLE],
)


def get_label_scheme(x: DistData, labels: List[str]):
    new_x = DistData()
    if len(labels) == 0:
        return new_x
    new_x.CopyFrom(x)
    assert x.type == "sf.table.individual"
    imeta = IndividualTable()
    assert x.meta.Unpack(imeta)

    new_meta = IndividualTable()
    names = []
    types = []

    for l, lt in zip(list(imeta.schema.labels), list(imeta.schema.label_types)):
        names.append(l)
        types.append(lt)

    for label in labels:
        if label not in names:
            raise CompEvalError(f"key {label} is not found as id or feature.")

    for n, t in zip(names, types):
        if n in labels:
            new_meta.schema.labels.append(n)
            new_meta.schema.label_types.append(t)

    logging.warning("label schemas: "+ str(list(imeta.schema.labels)))
    new_meta.schema.labels.extend(list(imeta.schema.labels))
    new_meta.schema.label_types.extend(list(imeta.schema.label_types))
    new_meta.line_count = imeta.line_count

    new_x.meta.Pack(new_meta)

    return new_x


@pir_comp.eval_fn
def pir_eval_fn(
        *,
        ctx,
        num_per_query,
        label_max_len,
        bucket_size,
        client_query_data_input,
        client_query_data_input_key,
        server_data_input,
        server_data_input_label,
        pir_output,
):
    import logging
    logging.warning("pir start...")

    client_path_format = extract_distdata_info(client_query_data_input)
    assert len(client_path_format) == 1
    client_party = list(client_path_format.keys())[0]
    server_path_format = extract_distdata_info(server_data_input)
    server_party = list(server_path_format.keys())[0]

    logging.warning("pir server_party: " + server_party)
    logging.warning("pir client_party: " + client_party)

    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))

    input_path = os.path.join(ctx.data_dir, server_path_format[server_party].uri)
    logging.warning("pir input_path: " + str(input_path))
    logging.warning("pir key column(s): " + str(client_query_data_input_key))
    logging.warning("pir label column(s): " + str(server_data_input_label))

    client_query_path = os.path.join(ctx.data_dir, client_path_format[client_party].uri)
    logging.warning("client query path: " + str(client_query_path))

    logging.warning(spu_config)

    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])
    logging.warning(spu_config)

    server_oprf_key = os.path.join(ctx.data_dir, "server_oprf_key")
    server_setup = os.path.join(ctx.data_dir, "server_setup")
    pir_result = os.path.join(ctx.data_dir, pir_output)

    logging.warning("pir server_oprf_key: " + str(server_oprf_key))
    logging.warning("pir server_setup: " + str(server_setup))
    logging.warning("pir pir_result: " + str(pir_result))

    with ctx.tracer.trace_running():
        spu.pir_setup(
            server=server_party,
            input_path=input_path,
            key_columns=client_query_data_input_key,
            label_columns=server_data_input_label,
            oprf_key_path=server_oprf_key,
            setup_path=server_setup,  # 中间结果存储路径
            num_per_query=num_per_query,
            label_max_len=label_max_len,
            bucket_size=bucket_size
        )

        spu.pir_query(
            server=server_party,
            client=client_party,
            server_setup_path=server_setup,
            client_key_columns=client_query_data_input_key,
            client_input_path=client_query_path,
            client_output_path=pir_result,
        )

    # client_meta = IndividualTable()
    # client_query_data_input.meta.Unpack(client_meta)
    #
    # server_meta = IndividualTable()
    # server_data_input.meta.Unpack(server_meta)
    #
    # logging.warning("client_meta: ", client_meta)
    # logging.warning("server_meta: ", server_meta)

    output_db = DistData(
        name=pir_output,
        type=str(DistDataType.VERTICAL_TABLE),
        system_info=client_query_data_input.system_info,
        data_refs=[
            DistData.DataRef(
                uri=pir_output,
                party=client_party,
                format="csv",
            ),
        ],
    )

    output_db = merge_individuals_to_vtable(
        [
            client_query_data_input,
            get_label_scheme(server_data_input, server_data_input_label),
        ],
        output_db,
    )

    return {"pir_output": output_db}
