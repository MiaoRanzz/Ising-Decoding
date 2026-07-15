目前任务：将 lqcloud 在实体硬件上检测 d=3 表面码的算法接入 nvidia ising decoder 中，在实体硬件上验证 ising decoder 的有效性。

lqcloud 技术路线（相关代码位于 `my_file/lqcloud/lqcloud_d3_surface_code` 中）：
- 根据表面码设计图（实验中使用 memory-Z circuit），写出 `build_cloud_circuit`，并构造实例 `qc = circuits.build_cloud_circuit(ini_state=ini_state, cycle=cycle)`。
- 将实例上传量子云平台执行（实验中使用 QZ01-surface_code），得到每一轮 stabilizer 的测量结果。
- 在同样的设计图下，构造 Stim 线路 `build_stim_circuit`，并创建实例。
- 将 Stim 线路和之前的测量结果塞给 pymatching 进行纠错。纠错前有一个 Raw logical error rate，纠错后得到一个预测的 Corrected logical error rate。
- 比较两种 logical error rate，得出结论：量子纠错有用。

英伟达 ising decoder inference 的运作流程：
- 在终端输入 `WORKFLOW=inference bash code/scripts/local_run.sh`。
- 调用 `code/workflows/run.py` 中 `run(cfg: DictConfig)`；调用 `run_surface(cfg: DictConfig)`；调用 `code/evaluation/inference.py` 中 `run_inference(model, device, dist, cfg)`；调用 `code/evaluation/logical_error_rate.py` 中 `count_logical_errors_with_errorbar(model, device, dist, cfg)`；调用 `run_inference_and_decode_pre_decoder_memory(model, device, dist, cfg)`。
- 再往后调用一些生成测试数据集的东西。

现在希望：使用 lqcloud 的量子云平台在真实硬件上跑一遍表面码 memory-z circuit，得到真实硬件上的 measurement 结果，再将 measurement 结果和相应的 stim circuit 塞给 ising decoder 的 inference，从而在真实硬件上验证 predecoder + pymatching 的作用。

你需要：先阅读整个工程，理解其架构（重点关注 ising decoder 的 inference 部分和 `my_file/lqcloud/lqcloud_d3_surface_code` 目前的架构）将 “将 Stim 线路和之前的测量结果塞给 pymatching 进行纠错” 这一步替换成 “将 Stim 线路和之前的测量结果塞给 ising decoder 进行纠错”。为此我设计的技术路径是：
- 依然使用 `code/scripts/local_run.sh` 跑整个工程，但此时的终端输入是 `WORKFLOW=integrate_to_nvidia CONFIG_NAME=config_lqcloud bash code/scripts/local_run.sh`，也就是使用 `conf/config_lqcloud.yaml` 生成 cfg（并且这个文件你是可以根据需要随便改的），然后依然调用 `run(cfg: DictConfig)` 和 `run_surface(cfg: DictConfig)`，触发 `cfg.workflow.task == "integrate_to_nvidia"` 分支。
- 从这个分支往下，遇到的所有函数可根据需要进行修改，具体方法是：比如，在 `run_inference_and_decode_pre_decoder_memory(model, device, dist, cfg)` 下面添加一个 `run_inference_and_decode_pre_decoder_memory_modified(model, device, dist, cfg)`，里面随便改。总之不要破坏原有 inference 架构。
- 最主要的一点：将 stim circuit 和 measurement 结果改成 ising decoder inference 能够接受的 input 形式，进行 inference。

关于 stim circuit：在 `my_file/lqcloud/lqcloud_d3_surface_code` 的工程中，在 `my_file/lqcloud/lqcloud_d3_surface_code/circuits.py` 中生成了相应的 stim circuit 和与之完全等价的 lqcloud circuit；而在 nvidia 的框架中，或许是在 `code/qec` 里面有生成 stim 的方法，我还没仔细看。请比较这两种方法目前是否适配；如果不适配，请决定应该用哪种生成方法所需的改动较少，并按照相应思路写代码。

关于 measurement：我已经在 `my_file/lqcloud/lqcloud_d3_surface_code/measurement.log` 中使用 `print(result.get_memory())` 打印了measurement的结果，是一个形如 ['110000001111000011000000000000000011000011110000011100001000000011000000010000100', '010110001111000010101000000010000101000011111000100000011000100010111001011111111'] 形式的列表。列表的项数是总的实验轮次，每一轮实验的 measurement 结果是一串 0/1，在 d=3 表面码（我主要需要实现的情况）下，一共是 81 个 0/1 ，因为一轮实验有九个 round，每一个 round 测 8 个 measurement qubit，最后一个 round 结束后测了 9 个 data qubit，因此一共是 9\times 8 + 9 = 81 个 0/1。你需要 parse 这个列表，改成 nvidia 解码器能够处理的格式。

请写相应代码。写完之后向我解释具体发生了什么，顺便也解释一下 ising decoder inference 原本的架构。