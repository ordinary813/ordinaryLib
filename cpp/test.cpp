#include <iostream>
#include <string>

void convertToASCII(std::string s)
{
    for (int i = 0; i < s.length(); i++)
    {
        std::cout << (int)s[i]<< std::endl;
    }
}

std::string convertToBinary(int num)
{
    if(num == 0)
        return 0;
    
    std::string binary;
    while(num != 0)
    {
        binary = std::to_string(num % 2) + binary;
        num /= 2;
    }
    return binary;
}

int main()
{
    int num;
    std::cout << "Enter number:";
    std::cin >> num;
    std::string binary = convertToBinary(num);

    std::cout << "Binary number:" << binary << std::endl;
    std::string bin2 = "111";
    binary += bin2;
    std::cout << "Binary number2:" << binary << std::endl;
    return 0;
}
