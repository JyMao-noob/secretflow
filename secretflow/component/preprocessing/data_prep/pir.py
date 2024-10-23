import os

from secretflow.component.component import (
    CompEvalError,
    Component,
    IoType,
)
from secretflow.component.data_utils import (
    DistDataType,
)
from secretflow.device.device.spu import SPU

# 声明组件
pir_comp = Component(
    "pir",
    domain="data_prep",
    version="0.0.1",
    desc="PIR between two parties.",
)
pir_comp.str_attr(
    name="client_node_name",
    desc="Which party is pir client",
    is_list=False,
    is_optional=False,
)

pir_comp.str_attr(
    name="server_node_name",
    desc="Which party is pir server",
    is_list=False,
    is_optional=False,
)


pir_comp.str_attr(
    name="client_query_data_path",
    desc="Client's query input path",
    is_list=False,
    is_optional=False,
)

pir_comp.str_attr(
    name="server_data_path",
    desc="Server's CSV file path. comma separated and contains header.",
    is_list=False,
    is_optional=False,
)

pir_comp.str_attr(
    name="key_columns",
    desc="Column(s) used as pir key",
    is_list=False,
    is_optional=False,
)

pir_comp.str_attr(
    name="label_columns",
    desc="Column(s) used as pir label",
    is_list=False,
    is_optional=False,
)

pir_comp.io(
    io_type=IoType.OUTPUT,
    name="pir_output",
    desc="Output vertical table",
    types=[DistDataType.VERTICAL_TABLE],
)

pir_comp.io(
    io_type=IoType.INPUT,
    name="input_data",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    col_params=None,
)


@pir_comp.eval_fn
def pir_eval_fn(
        *,
        ctx,
        client_node_name,
        server_node_name,
        key_columns,
        label_columns,
        client_query_data_path,
        server_data_path,
        input_data,
        pir_output,
):
    import logging
    logging.warning("pir start...")
    logging.warning("pir client_node_name: " + client_node_name)
    logging.warning("pir server_node_name: " + server_node_name)
    logging.warning("pir key_columns: " + key_columns)
    logging.warning("pir client_query_data_path: " + client_query_data_path)
    logging.warning("pir server_data_path: " + server_data_path)


    receiver_party = server_node_name
    sender_party = client_node_name
    logging.warning("pir receiver_party: " + receiver_party)
    logging.warning("pir sender_party: " + sender_party)

    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))

    input_path = os.path.join(ctx.data_dir, server_data_path)

    logging.warning("pir input_path: " + str(input_path))


    logging.warning(spu_config)

    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])
    logging.warning(spu_config)

    uri = {
        receiver_party: server_data_path,
    }

    # with ctx.tracer.trace_io():
    #     download_files(ctx, uri, input_path)

    server_oprf_key = os.path.join(ctx.data_dir, "server_oprf_key")
    server_setup = os.path.join(ctx.data_dir, "server_setup")
    pir_result = os.path.join(ctx.data_dir, pir_output)

    logging.warning("pir server_oprf_key: " + str(server_oprf_key))
    logging.warning("pir server_setup: " + str(server_setup))
    logging.warning("pir pir_result: " + str(pir_result))


    with ctx.tracer.trace_running():
        spu.pir_setup(
            server=server_node_name,         # 服务方node_name
            input_path=input_path,           # 服务方数据路径
            key_columns=key_columns,         # key列
            label_columns=label_columns,     # label列
            oprf_key_path=server_oprf_key,   # 服务方 secret key 路径
            setup_path=server_setup,         # 中间结果存储路径
            num_per_query=2,                 # 服务方每次处理的query数量
            label_max_len=20,                # label长度限制(byte)
            bucket_size=1000000              # 不可区分度,通常用百万级
        )

        spu.pir_query(
            server=server_node_name,         # 服务方node_name
            client=client_node_name,         # 查询方node_name
            server_setup_path=server_setup,  # 服务方 中间结果存储路径
            client_key_columns=key_columns,  # 查询方 key列
            client_input_path=client_query_data_path, # 查询方数据路径
            client_output_path=pir_result,   # 结果路径
        )













