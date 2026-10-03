#include "oxbot/mini_json.hpp"
#include <cassert>
#include <iostream>
#include <limits>

int main() {
    using namespace oxbot::json;
    assert(parse("1.5").as_int(-1) == -1);
    assert(parse("999999999999999999999").as_int(-1) == -1);
    assert(parse("-12").as_int() == -12);
    assert(parse("\"\\u63bc\\u86cb\\ud83c\\udccf\"").string == "掼蛋🃏");
    const auto sample = parse("{\"requests\":[{\"stage\":\"play\",\"history\":[[],{},[],{}]}],\"responses\":[],\"data\":null}");
    assert(dump(parse(dump(sample))) == dump(sample));
    for (const char* bad : {"01", "NaN", "1e309", "1e-", "[1,]", "{\"a\":1,\"a\":2}",
                                   "\"\\ud800\"", "\"\\udc00\"", "{\"x\":true} junk"}) {
        bool rejected = false;
        try { parse(bad); } catch (const ParseError&) { rejected = true; }
        assert(rejected);
    }
    bool rejected = false;
    try { parse(std::string(130, '[') + "0" + std::string(130, ']')); }
    catch (const ParseError&) { rejected = true; }
    assert(rejected);
    std::cout << "json tests passed\n";
}
