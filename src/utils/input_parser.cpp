#include "input_parser.hpp"

InputParser::InputParser(int &argc, char **argv) {
  for (int i = 1; i < argc; ++i) _tokens.push_back(string(argv[i]));
}

const string& InputParser::getArg(const int& i) const {
  return _tokens[i];
}

const std::string& InputParser::getCmdOption(const std::string &option) const {
  vector<string>::const_iterator itr;
  itr = std::find(_tokens.begin(), _tokens.end(), option);
  if (itr != _tokens.end() && ++itr != _tokens.end()) {
    return *itr;
  }
  static const string empty_string("");
  return empty_string;
}

bool InputParser::cmdOptionExists(const std::string &option) const {
  return std::find(_tokens.begin(), _tokens.end(), option) !=_tokens.end();
}
