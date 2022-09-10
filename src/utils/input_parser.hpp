#pragma once

#include <algorithm>
#include <vector>

using std::string;
using std::vector;

class InputParser {
  
  private:
    std::vector<std::string> _tokens;

  public:
    InputParser(int &argc, char **argv);

    const string& getArg(const int& i) const;
    const string& getCmdOption(const string& option) const;
    bool cmdOptionExists(const string& option) const;

};
