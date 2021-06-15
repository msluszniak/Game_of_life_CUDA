#include "inputer.h"

char get_mode() {
    std::cout << "Welcome to the game of life!" << std::endl;
    std::cout << "Please choose f if you want to load data from file or press m to input data manually." << std::endl;
input:
    char mode = getchar();
    if (mode != 'f' && mode != 'm') {
        std::cout << "You choose wrong mode: " << mode << ". Please use f for file or m for manually." << std::endl;
        goto input;
    }
    return mode;
}

void input_file(inputer_data* input) {
    std::cout << "Now put the name of the file with size and start positions" << std::endl;
    std::string file_name;
    std::cin >> file_name;

    std::ifstream infile(file_name.c_str());
    if (!infile) {
        std::cout << "Cannot open file correctly" << std::endl;
    }
    if (!(infile >> input->num_threads)) {
        std::cout << input->num_threads << std::endl;
        std::cout << "Wrong file format, file must include number of threads" << std::endl;
        exit(1);
    }
    if (!(infile >> input->image_width >> input->image_height)) {
        std::cout << "Wrong file format, file must include width and height" << std::endl;
        exit(1);
    }
    input->total_size = input->image_width * input->image_height;
    input->host_map = new bool[input->total_size];
    input->host_next_state = new bool[input->total_size];
    for (u_int i = 0; i < input->total_size; ++i)
        input->host_map[i] = input->host_next_state[i] = false;
    u_int num_start_alive;
    if (!(infile >> num_start_alive)) {
        std::cout << "Wrong file format, file must include number of alive cells at the beginning" << std::endl;
        exit(1);
    }
    u_int x_pos, y_pos;
    u_int num_loaded = 0;
    while (infile >> x_pos >> y_pos) {
        if (x_pos >= input->image_width)
            x_pos = x_pos % input->image_width;
        if (y_pos >= input->image_height)
            y_pos = y_pos % input->image_height;
        num_loaded += 2;
        input->host_map[x_pos * input->image_width + y_pos] = true;
    }
    if (num_loaded < 2 * num_start_alive) {
        std::cout << "File does not contains enough coords" << std::endl;
        delete[] input->host_map;
        delete[] input->host_next_state;
        exit(1);
    }
    infile.close();
}

void input_manual(inputer_data* input) {
    std::cout << "Please input number of threads" << std::endl;
    std::cin >> input->num_threads;
    std::cout << "Now, please input width and length" << std::endl;
    std::cin >> input->image_width >> input->image_height;
    input->total_size = input->image_width * input->image_height;
    input->host_map = new bool[input->total_size];
    input->host_next_state = new bool[input->total_size];
    for (u_int i = 0; i < input->total_size; ++i)
        input->host_map[i] = input->host_next_state[i] = false;
    u_int counter = 0;
    u_int x_pos, y_pos;
    u_int num_start_alive;
    std::cout << "Input number of alive cells at the beginning" << std::endl;
    std::cin >> num_start_alive;
    std::cout << "Input starting data" << std::endl;
    while (counter != num_start_alive) {
        std::cin >> x_pos >> y_pos;
        if (x_pos >= input->image_width)
            x_pos = x_pos % input->image_width;
        if (y_pos >= input->image_height)
            y_pos = y_pos % input->image_height;
        counter++;
        input->host_map[x_pos * input->image_width + y_pos] = true;
    }
}

inputer_data input_wrapper() {
    inputer_data input;
    char mode = get_mode();
    if (mode == 'f')
        input_file(&input);
    else if (mode == 'm')
        input_manual(&input);
    else {
        std::cout << "Something goes wrong" << std::endl;
    }
    return input;
}