#ifndef TEMP_DIR_H
#define TEMP_DIR_H

#include <string>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <cstring>

namespace test_helpers {

// RAII wrapper for temporary directory
class TempDir {
public:
    TempDir(const std::string& prefix = "shepherd_test_") {
        std::string tmpl = "/tmp/" + prefix + "XXXXXX";
        char* path = mkdtemp(const_cast<char*>(tmpl.c_str()));
        if (path) {
            path_ = path;
        }
    }

    ~TempDir() {
        if (!path_.empty()) {
            remove_recursive(path_);
        }
    }

    // No copy
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    // Move OK
    TempDir(TempDir&& other) noexcept : path_(std::move(other.path_)) {
        other.path_.clear();
    }

    TempDir& operator=(TempDir&& other) noexcept {
        if (this != &other) {
            if (!path_.empty()) {
                remove_recursive(path_);
            }
            path_ = std::move(other.path_);
            other.path_.clear();
        }
        return *this;
    }

    const std::string& path() const { return path_; }

    std::string file_path(const std::string& name) const {
        return path_ + "/" + name;
    }

    bool valid() const { return !path_.empty(); }

private:
    std::string path_;

    static void remove_recursive(const std::string& path) {
        DIR* dir = opendir(path.c_str());
        if (dir) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr) {
                if (strcmp(entry->d_name, ".") == 0 ||
                    strcmp(entry->d_name, "..") == 0) {
                    continue;
                }

                std::string full_path = path + "/" + entry->d_name;
                struct stat st;
                if (stat(full_path.c_str(), &st) == 0) {
                    if (S_ISDIR(st.st_mode)) {
                        remove_recursive(full_path);
                    } else {
                        unlink(full_path.c_str());
                    }
                }
            }
            closedir(dir);
        }
        rmdir(path.c_str());
    }
};

} // namespace test_helpers

#endif // TEMP_DIR_H
