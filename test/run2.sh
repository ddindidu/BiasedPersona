python main.py --persona_category 'Baseline' --target_category 'Age' --model meta-llama/Llama-2-13b-chat-hf
python main.py --persona_category 'Baseline' --target_category 'Race_ethnicity' --model meta-llama/Llama-2-13b-chat-hf


conda activate bias
python main.py --persona_category 'Race_ethnicity' --target_category 'Race_ethnicity' --model meta-llama/Llama-2-7b-chat-hf --instruction_start 1 --instruction_end 2
python main.py --persona_category 'Race_ethnicity' --target_category 'Race_ethnicity' --model meta-llama/Llama-2-7b-chat-hf --instruction_start 2 --instruction_end 3
python main.py --persona_category 'Race_ethnicity' --target_category 'Race_ethnicity' --model meta-llama/Llama-2-13b-chat-hf --instruction_start 1 --instruction_end 2
python main.py --persona_category 'Race_ethnicity' --target_category 'Race_ethnicity' --model meta-llama/Llama-2-13b-chat-hf --instruction_start 2 --instruction_end 3
python main.py --persona_category 'Race_ethnicity' --target_category 'Race_ethnicity' --model meta-llama/Llama-2-70b-chat-hf --instruction_start 1 --instruction_end 2
python main.py --persona_category 'Race_ethnicity' --target_category 'Race_ethnicity' --model meta-llama/Llama-2-70b-chat-hf --instruction_start 2 --instruction_end 3

python main.py --persona_category 'SES' --target_category 'SES' --model meta-llama/Llama-2-7b-chat-hf --instruction_start 1 --instruction_end 2
python main.py --persona_category 'SES' --target_category 'SES' --model meta-llama/Llama-2-7b-chat-hf --instruction_start 2 --instruction_end 3
python main.py --persona_category 'SES' --target_category 'SES' --model meta-llama/Llama-2-13b-chat-hf --instruction_start 1 --instruction_end 2
python main.py --persona_category 'SES' --target_category 'SES' --model meta-llama/Llama-2-13b-chat-hf --instruction_start 2 --instruction_end 3
python main.py --persona_category 'SES' --target_category 'SES' --model meta-llama/Llama-2-70b-chat-hf --instruction_start 1 --instruction_end 2
python main.py --persona_category 'SES' --target_category 'SES' --model meta-llama/Llama-2-70b-chat-hf --instruction_start 2 --instruction_end 3