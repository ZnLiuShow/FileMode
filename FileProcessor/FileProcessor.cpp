// FileProcessor.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <filesystem>
#include <string>
#include <unordered_map>

namespace fs = std::filesystem;

void renameFilesInSequence(const fs::path& directoryPath) {
    std::vector<std::pair<int, fs::path>> fileList;
    std::unordered_map<int, int> renameMap;

    // 遍历目录，收集所有文件及其编号
    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        if (fs::is_regular_file(entry)) {
            std::string filename = entry.path().filename().string();
            size_t dotPos = filename.find_last_of('.');
            if (dotPos != std::string::npos) {
                std::string namePart = filename.substr(0, dotPos);
                std::string extPart = filename.substr(dotPos);
                int number = 0;

                // 尝试提取数字部分
                try {
                    number = std::stoi(namePart);
                    fileList.push_back({ number, entry.path() });
                }
                catch (std::invalid_argument&) {
                    continue; // 如果转换失败，则跳过此文件
                }
            }
        }
    }

    // 对文件列表排序
    std::sort(fileList.begin(), fileList.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    // 构建重命名映射
    int newNumber = fileList.front().first;
    for (const auto& [oldNumber, _] : fileList) {
        renameMap[oldNumber] = newNumber++;
    }

    // 执行重命名操作
    for (const auto& [oldNumber, filePath] : fileList) {
        std::string filename = filePath.filename().string();
        size_t dotPos = filename.find_last_of('.');
        std::string extPart = filename.substr(dotPos);
        std::string newName = std::to_string(renameMap[oldNumber]) + extPart;
        fs::path newPath = filePath;
        newPath.replace_filename(newName);

        std::error_code ec;
        fs::rename(filePath, newPath, ec);
        if (ec) {
            std::cout << "Failed to rename file " << filePath << " to " << newPath << ": " << ec.message() << "\n";
        }
        else {
            std::cout << "Renamed " << filePath << " to " << newPath << "\n";
        }
    }
}

int main(int argc, char* argv[])
{
	if (argc < 2)
		return 0;
	std::string path = argv[1];
    renameFilesInSequence(path.c_str());
    return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
