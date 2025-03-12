from .logger import logger


def stream_response(chain, state):
    result = []

    for chunk in chain.stream(state):
        result.append(chunk)
        # print(chunk, end="", flush=True)
    return "".join(result)


def invoke_graph(graph, state, config, stream):
    logger.info(
        "Invoking graph with state: {}, config: {}, stream: {}",
        state.keys(),
        config,
        stream,
    )
    if stream:
        for s in graph.stream(state, config=config):
            logger.debug(s)

        return graph.get_state(config).values
    else:
        result = graph.invoke(state, config=config)
        return result
