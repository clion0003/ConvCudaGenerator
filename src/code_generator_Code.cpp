inline std::string GenFillW(std::string src)
{
    std::stringstream ins;
    ins << "fillw.f32 " << src << ";" << "\n";
    return ins.str();
}

inline std::string GenShiftCol(std::string dst, unsigned int dst_offset, unsigned int dst_stride, 
    std::string src, unsigned int src_offset, unsigned int src_stride)
{
    std::stringstream ins;
    ins << "shiftcol.f32 " << dst << ", " << dst_offset << ", " << src << ", " << src_offset << ", " << src_stride << ", " << dst_stride << ";" << "\n";
    return ins.str();
}

inline std::string GenShiftFull(std::string dst, unsigned int dst_offset, unsigned int dst_stride, 
    std::string src, unsigned int src_offset, unsigned int src_stride)
{
    std::stringstream ins;
    ins << "shiftfull.f32 " << dst << ", " << dst_offset << ", " << src << ", " << src_offset << ", " << src_stride << ", " << dst_stride << ";" << "\n";
    return ins.str();
}

std::string GenCNN3x3ComputingRow(unsigned int output_row_size, unsigned int kernel_num,
	const std::vector<std::string> input_reg_address, unsigned int input_reg_offset, unsigned int input_reg_stride,
	const std::vector<std::string> kernel_reg_address,
	const std::vector<std::string> output_reg_address, unsigned int output_reg_offset, unsigned int output_reg_stride, unsigned int output_reg_mode)
{
	using std::string;
	using std::vector;

	string return_str = "";
	if (output_reg_mode == 1)
	{
		return_str += GenShiftFull(output_reg_address[0], output_reg_offset, output_reg_stride,
			input_reg_address[0], input_reg_offset, input_reg_stride);

		unsigned int output_reg_address_index = 0;
		unsigned int output_reg_offset_tmp = output_reg_offset;
		unsigned int input_reg_address_index = 0;
		unsigned int input_reg_offset_tmp = input_reg_offset + 2;

		for (unsigned int i = 0; i < output_row_size - 1; i++)
		{
			input_reg_offset_tmp += 1;
			if (input_reg_offset_tmp >= 36)
			{
				input_reg_offset_tmp -= 36;
				input_reg_address_index++;
			}

			output_reg_offset_tmp += 1;
			if (output_reg_offset_tmp >= 36)
			{
				output_reg_offset_tmp -= 36;
				output_reg_address_index++;
			}

			return_str += GenShiftCol(output_reg_address[output_reg_address_index], output_reg_offset_tmp, output_reg_stride,
				input_reg_address[input_reg_address_index], input_reg_offset_tmp, input_reg_stride);
		}

		return return_str;
	}
    else
    {
        return "";
    }
}

std::string LdStCNNChannelRow(unsigned int row_size, std::string dram_name, unsigned int dram_offset,
	std::string tid_reg_address, std::string dram_address_reg_name, std::string cal_result_reg_address,
	const std::vector<std::string> reg_address, unsigned int reg_offset, unsigned int instruction_mode)
{
	std::stringstream return_ss;
	//return_ss << "ld.param.u64 " << dram_address_reg_name << ", " << "[" << dram_name << "]" << ";" << "\n";
	//return_ss << "add.u64 " << cal_result_reg_address << ", " << tid_reg_address << ", " << dram_address_reg_name << ";" << "\n";
	int row_size_tmp = row_size;
	int dram_offset_tmp = dram_offset;
	int reg_offset_tmp = reg_offset;
	int reg_address_index = 0;
	while (row_size_tmp > 0)
	{
		int ld_length;
		int remain_regs = 36 - reg_offset_tmp;
		//dst_reg_offset_tmp = (dst_reg_offset_tmp + 3) % 36;
		ld_length = remain_regs < row_size_tmp ? remain_regs : row_size_tmp;
		row_size_tmp -= remain_regs;
		if (instruction_mode == 1) {
			return_ss << "ldc.global.f32 " << reg_address[reg_address_index] << ", "
				<< "[" << cal_result_reg_address << "+" << dram_offset_tmp << "]" << ", " << reg_offset_tmp << ", " << ld_length << ";" << "\n";
		}
		else if(instruction_mode == 2){
			return_ss << "stc.global.f32 " << "[" << cal_result_reg_address << "+" << dram_offset_tmp << "]" << ", "
				<< reg_address[reg_address_index] << ", " << reg_offset_tmp << ", " << ld_length << ";" << "\n";
		}
		reg_offset_tmp = 0;
		dram_offset_tmp += ld_length * 4;
		reg_address_index++;
	}
	return return_ss.str();
}


std::vector<std::string> GenRegNames(std::string reg_name_prefix, std::vector<unsigned int> reg_no) 
{
	std::vector<std::string> reg_names;
	for(auto number : reg_no)
	{
		std::stringstream reg_names_ss;
		reg_names_ss << "%" << reg_name_prefix << number;
		reg_names.push_back(reg_names_ss.str());
	}
	return reg_names;
}

std::string GenCNNComputingIterationVersion(unsigned int input_channel_num, unsigned int input_kernel_num, unsigned int output_row_size, unsigned int output_col_size,
	std::vector<std::string> iteration_reg_address, std::vector<std::string> address_cal_reg_address,
	std::string input_dram_address, std::string input_reg_prefix, unsigned int input_reg_start, unsigned int input_reg_end, unsigned int input_reg_start_offset,
	std::string kernel_dram_address, std::vector<std::string> kernel_reg_address,
	std::string output_dram_address, std::string output_reg_prefix, unsigned int output_reg_start, unsigned int output_reg_end, unsigned int output_reg_start_offset,
	std::string kernel_bias_dram_address, std::string bias_value_reg[4])
{
	static unsigned int output_line_num_periter = 2;
	static unsigned int cal_kernel_num_periter = 1;
	static unsigned int pre_fetch_line_gap = 1;

	unsigned int line_iteration_remind = input_channel_num % (pre_fetch_line_gap + 1);
	unsigned int line_iteration = input_channel_num / (pre_fetch_line_gap + 1);

	using std::endl;
	std::stringstream of;
	std::string input_dram_tmp = address_cal_reg_address[0];
	std::string kernel_dram_tmp = address_cal_reg_address[1];
	std::string output_dram_tmp = address_cal_reg_address[2];
	std::string kernel_dram_tmp2 = address_cal_reg_address[3];
	std::string kernel_bias_dram_tmp = address_cal_reg_address[4];
	std::string input_dram_tmp2 = address_cal_reg_address[5];
	std::string output_dram_tmp2 = address_cal_reg_address[6];

	std::string channel_num_tmp = iteration_reg_address[0];
	std::string compare_num1 = iteration_reg_address[1];
	std::string kernel_num_tmp = iteration_reg_address[2];
	std::string compare_num2 = iteration_reg_address[3];
	std::string compare_num3 = iteration_reg_address[4];
	std::string line_num = iteration_reg_address[5];

	int register_need_eachrow = (output_row_size + 35 + 33) / 36;
	std::vector<std::vector<std::vector<std::string>>> input_reg_address_blocks;
	std::vector<std::vector<std::vector<std::string>>> output_reg_address_blocks;
	std::vector<unsigned int> input_reg_offset;
	std::vector<unsigned int> output_reg_offset;

	int input_row_size = output_row_size + 2;
	int input_col_size = output_col_size + 2;

	{
		unsigned int input_reg_tmp = input_reg_start;
		for (int input_reg_block_num = 0; input_reg_block_num < 1 + pre_fetch_line_gap; input_reg_block_num++) 
		{
			std::vector<std::vector<std::string>> input_reg_address_block;
			for(int row_num = 0; row_num < 2 + output_line_num_periter ;row_num ++)
			{
				std::vector<unsigned int> freg;
				for (unsigned int t = 0; t < register_need_eachrow; t++)
				{
					freg.push_back(input_reg_tmp + t);
				}
				auto freg_name = GenRegNames(input_reg_prefix, freg);
				input_reg_address_block.push_back(freg_name);
				input_reg_tmp += register_need_eachrow;
			}
			input_reg_address_blocks.push_back(input_reg_address_block);
		}

		for (int row_num = 0; row_num < 2 + output_line_num_periter; row_num++)
		{
			input_reg_offset.push_back(3 * row_num);
		}
	}

	{
		unsigned int output_reg_tmp = output_reg_start;
		for (int output_reg_block_num = 0; output_reg_block_num < output_line_num_periter; output_reg_block_num++)
		{
			std::vector<std::vector<std::string>> output_reg_address_block;
			for (int row_num = 0; row_num < 4; row_num++)
			{
				std::vector<unsigned int> freg;
				for (unsigned int t = 0; t < register_need_eachrow; t++)
				{
					freg.push_back(output_reg_tmp + t);
				}
				auto freg_name = GenRegNames(output_reg_prefix, freg);
				output_reg_address_block.push_back(freg_name);
				output_reg_tmp += register_need_eachrow;
			}
			output_reg_address_blocks.push_back(output_reg_address_block);
		}

		for (int row_num = 0; row_num < 4; row_num++)
		{
			output_reg_offset.push_back(20 + row_num);
		}
	}


	of << "mov.s32 " << compare_num2 << ", " << input_kernel_num / ( 4 * cal_kernel_num_periter ) << ";" << endl;
	of << "mov.s32 " << compare_num1 << ", " << line_iteration << ";" << endl;
	of << "mov.s32 " << compare_num3 << ", " << output_row_size / output_line_num_periter << ";" << endl;

	of << "mov.u64 " << output_dram_tmp << ", " << output_dram_address << ";" << endl;
	of << "mov.u64 " << kernel_dram_tmp << ", " << kernel_dram_address << ";" << endl;
	of << "mov.u64 " << input_dram_tmp << ", " << input_dram_address << ";" << endl;
	of << "mov.f32 %f1 , 0f00000000;"  << endl;

	{
		of << "mov.s32 " << line_num << ", 0;" << endl;
		of << "$Lt_0:" << endl;

		of << "mov.u64 " << kernel_bias_dram_tmp << ", " << kernel_bias_dram_address << ";" << endl;
		of << "mov.u64 " << kernel_dram_tmp << ", " << kernel_dram_address << ";" << endl;
		{

			of << "mov.u64 " << output_dram_tmp2 << ", " << output_dram_tmp << ";" << endl;
			//for (int kernel_n = 0; kernel_n < 4; kernel_n++) 
			//{
			//	for (int t = 0; t < register_need_eachrow; t++) 
			//	{
			//		//of << "mov.f32 %f" << output_reg_start + kernel_n * register_need_eachrow + t << ", 0f00000000;" << std::endl;
			//		of << "mov.f32 " << output_reg_address_blocks[0][kernel_n][t] << ", " << bias_value_reg[kernel_n] << ";" << std::endl;
			//	}
			//}

			of << "mov.s32 " << kernel_num_tmp << ", 0;" << endl;
			of << "$Lt_0_0:" << endl;

			of << "ld.global.f32 " << bias_value_reg[0] << ", [" << kernel_bias_dram_tmp << "+0];" << endl;
			of << "ld.global.f32 " << bias_value_reg[1] << ", [" << kernel_bias_dram_tmp << "+4];" << endl;
			of << "ld.global.f32 " << bias_value_reg[2] << ", [" << kernel_bias_dram_tmp << "+8];" << endl;
			of << "ld.global.f32 " << bias_value_reg[3] << ", [" << kernel_bias_dram_tmp << "+12];" << endl;

			for (int row_num = 0; row_num < output_line_num_periter; row_num++)
			{
				for (int output_num = 0; output_num < 4; output_num++)
				{
					for (int reg_num = 0; reg_num < register_need_eachrow; reg_num++) 
					{
						of << "mov.f32 " << output_reg_address_blocks[row_num][output_num][reg_num] << ", " 
							<< bias_value_reg[output_num] << ";" << std::endl;
					}
				}
			}


			//prefetch kernel
			for (int kernel_prefetch_num = 0; kernel_prefetch_num < pre_fetch_line_gap; kernel_prefetch_num ++) 
			{
				int kernel_offset = kernel_prefetch_num * input_kernel_num * 3 * 3 * 4;
				of << "ldc.global.f32 " << kernel_reg_address[kernel_prefetch_num] << ",["
					<< kernel_dram_tmp << "+" << kernel_offset << "], " << 0 << ", 36;" << endl;
			}

			//prefetch input
			for (int line_prefetch_num = 0; line_prefetch_num < pre_fetch_line_gap; line_prefetch_num++)
			{
				for (int row_num = 0; row_num < 2 + output_line_num_periter; row_num++)
				{
					of << LdStCNNChannelRow(input_row_size, "no matter", (input_row_size * row_num + line_prefetch_num * input_col_size * input_row_size) * 4,
						"no matter", "no matter", input_dram_tmp, input_reg_address_blocks[line_prefetch_num][row_num], input_reg_offset[row_num], 1);
				}
			}

			unsigned int input_reg_address_blocks_ptr = 0;
			{
				of << "add.u64 " << input_dram_tmp2 << ", " << input_dram_tmp << ", " << input_col_size * input_row_size * 4 * pre_fetch_line_gap << ";" << endl;
				of << "add.u64 " << kernel_dram_tmp2 << ", " << kernel_dram_tmp << ", " << 3 * 3 * input_kernel_num * pre_fetch_line_gap * 4 << ";" << endl;

				of << "mov.s32 " << channel_num_tmp << ", 0;" << endl;
				of << "$Lt_0_0_0:" << endl;

				unsigned int input_reg_address_blocks_ptr = 0;
				for (int iteration_num = 0; iteration_num < pre_fetch_line_gap + 1; iteration_num++)
				{
					unsigned int prefetch_block_num = (input_reg_address_blocks_ptr + pre_fetch_line_gap)%(pre_fetch_line_gap + 1);
					of << "ldc.global.f32 " << kernel_reg_address[prefetch_block_num] << ",["
						<< kernel_dram_tmp2 << "], " << 0 << ", 36;" << endl;
					for (int row_num = 0; row_num < 2 + output_line_num_periter; row_num++)
					{
						of << LdStCNNChannelRow(input_row_size, "no matter", (input_row_size * row_num) * 4,
							"no matter", "no matter", input_dram_tmp2, input_reg_address_blocks[prefetch_block_num][row_num], input_reg_offset[row_num], 1);
					}
					of << GenFillW(kernel_reg_address[input_reg_address_blocks_ptr]);
					for (int row_num = 0 ; row_num < output_line_num_periter; row_num++)
					of << GenCNN3x3ComputingRow(output_row_size, 4, input_reg_address_blocks[input_reg_address_blocks_ptr][row_num], input_reg_offset[row_num],
						register_need_eachrow, kernel_reg_address, output_reg_address_blocks[row_num][0], output_reg_offset[0], register_need_eachrow, 1);
					input_reg_address_blocks_ptr++;

					of << "add.u64 " << input_dram_tmp2 << ", " << input_dram_tmp2 << ", " << input_col_size * input_row_size * 4 << ";" << endl;
					of << "add.u64 " << kernel_dram_tmp2 << ", " << kernel_dram_tmp2 << ", " << 3 * 3 * input_kernel_num * 4 << ";" << endl;
				}

				of << "add.s32 " << channel_num_tmp << ", " << channel_num_tmp << ", " << 1 << ";" << endl;
				of << "setp.ne.s32 " << "%p1, " << channel_num_tmp << ", " << compare_num1 << ";" << endl;
				of << "@%p1 bra $Lt_0_0_0;" << endl;
			}

			for (int iteration_remind_index = 0; iteration_remind_index < line_iteration_remind; iteration_remind_index++)
			{
				of << GenFillW(kernel_reg_address[input_reg_address_blocks_ptr]);
				for (int row_num = 0; row_num < output_line_num_periter; row_num++)
					of << GenCNN3x3ComputingRow(output_row_size, 4, input_reg_address_blocks[input_reg_address_blocks_ptr][row_num], input_reg_offset[row_num],
						register_need_eachrow, kernel_reg_address, output_reg_address_blocks[row_num][0], output_reg_offset[0], register_need_eachrow, 1);
				input_reg_address_blocks_ptr++;

				of << "add.u64 " << input_dram_tmp2 << ", " << input_dram_tmp2 << ", " << input_col_size * input_row_size * 4 << ";" << endl;
				of << "add.u64 " << kernel_dram_tmp2 << ", " << kernel_dram_tmp2 << ", " << 3 * 3 * input_kernel_num * 4 << ";" << endl;
			}


			for (int kernel_n = 0; kernel_n < 4; kernel_n++)
			{
				for (int row_num = 0; row_num < output_line_num_periter; row_num++)
				{
					//of << LdStCNNChannelRowWithRelu(output_row_size, "no matter", 
					//	(output_row_size * output_col_size * kernel_n + row_num * output_row_size) *  4, 
					//	"no matter", "no matter", output_dram_tmp2, output_reg_address_blocks[row_num][kernel_n], 
					//	output_reg_offset[kernel_n], 2, row_num * 4 + kernel_n);
					of << LdStCNNChannelRow(output_row_size, "no matter", 
						(output_row_size * output_col_size * kernel_n + row_num * output_row_size) *  4, 
						"no matter", "no matter", output_dram_tmp2, output_reg_address_blocks[row_num][kernel_n], 
						output_reg_offset[kernel_n], 2);
					//of << "add.u64 " << output_dram_tmp2 << ", " << output_dram_tmp2 << ", " << output_row_size * 4 << ";" << endl;
				}
				//of << "add.u64 " << output_dram_tmp2 << ", " << output_dram_tmp2 << ", " << output_row_size * output_col_size * 4 - output_line_num_periter * output_row_size * 4 << ";" << endl; 
			}

			//of << "add.u64 " << kernel_dram_tmp << ", " << kernel_dram_tmp << ", " << 3 * 3 * input_channel_num * 4 * 4 << ";" << endl;
			//of << "add.u64 " << output_dram_tmp << ", " << output_dram_tmp << ", " << output_row_size * output_col_size * 4 * 4 << ";" << endl;
			//of << "add.u64 " << kernel_bias_dram_tmp << ", " << kernel_bias_dram_tmp << ", " << 16 << ";" << endl;

			of << "add.u64 " << output_dram_tmp2 << ", " << output_dram_tmp2 << ", " << output_row_size * output_col_size * 4 * 4 << ";" << endl;
			of << "add.u64 " << kernel_dram_tmp << ", " << kernel_dram_tmp << ", " << 3 * 3 * 4 * cal_kernel_num_periter * 4 << ";" << endl;
			of << "add.u64 " << kernel_bias_dram_tmp << ", " << kernel_bias_dram_tmp << ", " << 16 << ";" << endl;
			of << "add.s32 " << kernel_num_tmp << ", " << kernel_num_tmp << ", " << 1 << ";" << endl;
			of << "setp.ne.s32 " << "%p2, " << kernel_num_tmp << ", " << compare_num2 << ";" << endl;
			of << "@%p2 bra $Lt_0_0;" << endl;
		}

		of << "add.u64 " << output_dram_tmp << ", " << output_dram_tmp << ", " << output_row_size * output_line_num_periter * 4 << ";" << endl;
		of << "add.u64 " << input_dram_tmp << ", " << input_dram_tmp << ", " << input_row_size * output_line_num_periter * 4 << ";" << endl;

		of << "add.s32 " << line_num << ", " << line_num << ", " << 1 << ";" << endl;
		of << "setp.ne.s32 " << "%p3, " << line_num << ", " << compare_num3 << ";" << endl;
		of << "@%p3 bra $Lt_0;" << endl;
	}

	of << "ret;" << endl;

	return of.str();
}


std::string convgenerator(std::string input_param_name, std::string kernel_param_name, std::string output_param_name, std::string kernel_bias_name
                            int output_kernel_width, int kernel_num, int output_channel_num){
	std::stringstream of;

	using std::endl;

	of << "cvt.s32.u16 %r1, %tid.x;" << endl;
	of << "cvt.s64.s32 %rd1, %r1;" << endl;
	of << "mul.wide.s32 %rd2, %r1, 4;" << endl;
	of << "ld.param.u64 %rd3, [" << input_param_name << "];" << endl;
	of << "add.u64 %rd4, %rd3, %rd2;" << endl;
	of << "ld.param.u64 %rd5, [" << kernel_param_name << "];" << endl;
	of << "add.u64 %rd6, %rd5, %rd2;" << endl;
	of << "ld.param.u64 %rd7, [" << output_param_name << "];" << endl;
	of << "add.u64 %rd8, %rd7, %rd2;" << endl;

	of << "ld.param.u64 %rd9, [" << kernel_bias_name << "];" << endl;

	of << endl;
	
	//of << "mov.f32 %f1 0f00000000;" << endl;
	//{
	//	//set 0 to output_channel_area
	//	int size = 56 * 56 * 256;
	//	int times = size / 36;
	//	int remain = size % 36;
	//	for (int j = 0; j < times; j++)
	//	{
	//		int offset = 36 * j * 4;
	//		of << "st.global.f32 [%rd8+" << offset << "], %f1;" << endl;
	//	}
	//	of << "mov.u32 %r2, " << remain - 1 << ";" << endl;
	//	of << "setp.gt.s32 %p1, %r1, %r2" << endl;
	//	of << "@%p1 bra $Lt_0;" << endl;
	//	of << "st.global.f32 [%rd8+" << 36 * times * 4 << "], %f1;" << endl;
	//	of << "$Lt_0:" << endl;
	//}
	//of << endl;
	std::vector<std::string> iteration_reg_address, address_cal_reg_address;
	iteration_reg_address.push_back("%r18");
	iteration_reg_address.push_back("%r19");
    iteration_reg_address.push_back("%r20");
    iteration_reg_address.push_back("%r21");
	iteration_reg_address.push_back("%r22");
	iteration_reg_address.push_back("%r23");
	iteration_reg_address.push_back("%r24");
	address_cal_reg_address.push_back("%rd10");
	address_cal_reg_address.push_back("%rd11");
	address_cal_reg_address.push_back("%rd12");
	address_cal_reg_address.push_back("%rd13");
	address_cal_reg_address.push_back("%rd14");
	address_cal_reg_address.push_back("%rd15");
	address_cal_reg_address.push_back("%rd16");
	address_cal_reg_address.push_back("%rd17");


	std::string reg[4];
	for(int i=0;i<4;i++)reg[i] = "%f" + std::to_string(250+i);

	std::vector<std::string> kernel_reg; 
	for (int i = 0; i < 6; i++)
	{
		std::stringstream is;
		is << "%f" << i + 4;
		kernel_reg.push_back(is.str());
	}

	int input_channel_num = kernel_num;
	int input_channel_width = output_kernel_width;
	int output_channel_num = output_channel_num;

	of << GenCNNComputingIterationVersion(input_channel_num, output_channel_num, input_channel_width, input_channel_width, iteration_reg_address, address_cal_reg_address, "%rd4", "f", 10, 199, 0, "%rd6", kernel_reg, "%rd8", "f", 200, 249, 10, "%rd9", reg);
    return of.str();
}