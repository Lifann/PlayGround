#include <iostream>
#include <string>

union MyUnion {
	uint32_t SUCCESS;
	uint32_t FAIL;
	char INFO;
	uint64_t LONG_SUC;
};


int main() {
	MyUnion status;
	status.SUCCESS = 97;

	std::cout << status.INFO<< std::endl;
	return 0;
}
