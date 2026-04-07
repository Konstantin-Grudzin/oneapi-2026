#include "permutations_cxx.h"

#include <algorithm>
#include <unordered_map>

void Permutations(dictionary_t &dictionary)
{
    std::unordered_map<std::string, std::vector<std::string>> classes;

    for (dictionary_t::reverse_iterator it = dictionary.rbegin(); it != dictionary.rend(); ++it)
    {
        std::string signature = it->first;
        std::sort(signature.begin(), signature.end());
        classes[signature].push_back(it->first);
    }

    for (dictionary_t::iterator it = dictionary.begin(); it != dictionary.end(); ++it)
    {
        const std::string &current_word = it->first;
        std::vector<std::string> &permutations = it->second;

        std::string signature = current_word;
        std::sort(signature.begin(), signature.end());

        const std::vector<std::string> &same_class = classes[signature];

        permutations.clear();
        if (!same_class.empty())
        {
            permutations.reserve(same_class.size() - 1);
        }

        for (size_t i = 0; i < same_class.size(); ++i)
        {
            if (same_class[i] != current_word)
            {
                permutations.push_back(same_class[i]);
            }
        }
    }
}
