#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

const std::string color_of(int value)
{
	switch (value) {
		case -1:
			return "\e[107m";
		case 0:
			return "\e[100m";
		default:
			return "\e[40m";
	}
}

int main(int argc, char* argv[])
{
	if (argc != 2) return 1;

	std::ifstream dataset(argv[1]);
	if (!dataset.is_open()) return 1;

	std::string line;
	while (std::getline(dataset, line)) {
		std::stringstream line_stream(line);
		for (int y = 0; y < 9; ++y) {
			for (int x = 0; x < 7; ++x) {
				int value;
				line_stream >> value;
				line_stream.get();
				std::cout << color_of(value) << " " << "\e[m";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}
