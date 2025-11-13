
doit() {
	. ~/venv/bin/activate
	model=$1
	shift
	out=$1
	shift
	ctx=$1
	shift
	./build_engine.py $model --output_dir $out --max_seq_len $ctx --max_batch_size 1 --tp 1 --pp 3 $* | tee build.log
}
doit /home/steve/models/models--Skywork--MindLink-32B-0801/snapshots/1c102e4027b8f9788bec9587c4a78da8150639ba ~/models/MindLink-32B-0801_int8_engine 8192 $*
#doit /home/steve/models/MindLink-32B-0801_INT8 /home/steve/models/MindLink-32B-0801_INT8_engine 8192 $*
