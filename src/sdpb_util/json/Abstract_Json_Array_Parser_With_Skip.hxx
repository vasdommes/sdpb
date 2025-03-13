#pragma once

#include "Abstract_Json_Element_Parser.hxx"

#include <functional>

template <class TResult, class TElementParser>
class Abstract_Json_Array_Parser_With_Skip
    : public Abstract_Json_Element_Parser<TResult>
{
public:
  using element_type = typename TElementParser::value_type;
  using value_type = TResult;
  using base_type = Abstract_Json_Element_Parser<TResult>;

protected:
  ~Abstract_Json_Array_Parser_With_Skip() = default;

private:
  using SizeType = rapidjson::SizeType;
  using Ch = rapidjson::UTF8<>::Ch;

  std::function<bool(size_t index)> skip_element;

  enum State
  {
    Start,
    Inside,
    // TODO maybe remove End? We already have on_end() callbacks.
    End
  } state
    = Start;

  size_t index = 0;
  int array_level = 0;

  TElementParser element_parser;

public:
  template <class... TArgs>
  Abstract_Json_Array_Parser_With_Skip(
    bool skip, const std::function<void(TResult &&)> &on_parsed,
    const std::function<void()> &on_skipped,
    const std::function<bool(size_t index)> &skip_element,
    TArgs &&...element_parser_args);

private:
  virtual void on_element_parsed(element_type &&value, size_t index) = 0;
  virtual void on_element_skipped(size_t index) = 0;

public:
  void reset(const bool skip) override
  {
    this->skip = skip;
    element_parser.reset(this->skip_element(0));
    state = Start;
    index = 0;
    array_level = 0;
    clear_result();
  }

protected:
  virtual void clear_result() = 0;

// If we're parsing array elements, delegate to element parser.
// Otherwise, parse value by ourselves (e.g. call this_json_string() if we got JSON string instead of array)
#define ABSTRACT_JSON_ARRAY_ELEMENT_PARSER(func)                              \
  (state == Inside ? element_parser.func : this_##func)

public:
  bool json_null() final
  {
    return ABSTRACT_JSON_ARRAY_ELEMENT_PARSER(json_null());
  }
  bool json_bool(bool b) final
  {
    return ABSTRACT_JSON_ARRAY_ELEMENT_PARSER(json_bool(b));
  }
  bool json_int(int i) final
  {
    return ABSTRACT_JSON_ARRAY_ELEMENT_PARSER(json_int(i));
  }
  bool json_uint(unsigned i) final
  {
    return ABSTRACT_JSON_ARRAY_ELEMENT_PARSER(json_uint(i));
  }
  bool json_int64(int64_t i) final
  {
    return ABSTRACT_JSON_ARRAY_ELEMENT_PARSER(json_int64(i));
  }
  bool json_uint64(uint64_t i) final
  {
    return ABSTRACT_JSON_ARRAY_ELEMENT_PARSER(json_uint64(i));
  }
  bool json_double(double d) final
  {
    return ABSTRACT_JSON_ARRAY_ELEMENT_PARSER(json_double(d));
  }
  bool json_raw_number(const Ch *str, SizeType length, bool copy) final
  {
    return ABSTRACT_JSON_ARRAY_ELEMENT_PARSER(
      json_raw_number(str, length, copy));
  }
  bool json_string(const Ch *str, SizeType length, bool copy) final
  {
    return ABSTRACT_JSON_ARRAY_ELEMENT_PARSER(json_string(str, length, copy));
  }
  bool json_start_object() final
  {
    return ABSTRACT_JSON_ARRAY_ELEMENT_PARSER(json_start_object());
  }
  bool json_key(const Ch *str, SizeType length, bool copy) final
  {
    return ABSTRACT_JSON_ARRAY_ELEMENT_PARSER(json_key(str, length, copy));
  }
  bool json_end_object(SizeType memberCount) final
  {
    return ABSTRACT_JSON_ARRAY_ELEMENT_PARSER(json_end_object(memberCount));
  }
  bool json_start_array() final
  {
    ++array_level;
    switch(state)
      {
      case Start:
        // NB: we set element_parser.skip in constructor,
        // so this is not necessary here.
        // TODO: remove skip from constructors (set false by default)?
        // This would make code less verbose,
        // but feels somewhat less safe.
        element_parser.reset(this->skip_element(0));
        state = Inside;
        return true;
      case Inside:
        return ABSTRACT_JSON_ARRAY_ELEMENT_PARSER(json_start_array());
      default: return base_type::json_start_array();
      }
  }
  bool json_end_array(SizeType elementCount) final
  {
    --array_level;
    switch(state)
      {
        case Inside: {
          if(array_level == 0)
            {
              state = End;
              // TODO callbacks in on_end() can change state!
              // e.g. it always happens in case of nested arrays
              // Feels like spaghetti, need refactoring
              this->on_end();
              return true;
            }
          return ABSTRACT_JSON_ARRAY_ELEMENT_PARSER(
            json_end_array(elementCount));
        }
      default: return base_type::json_start_array();
      }
  }

protected:
  // This functions can be overridden in derived parsers,
  // For example, if we expect either array or string,
  // then we should implement this_json_string().
  // TODO: make a universal base class that can parse object, array, or simple value.

#define VIRTUAL_NOT_IMPLEMENTED(func)                                         \
  [[noreturn]] virtual func                                                   \
  {                                                                           \
    RUNTIME_ERROR("Not implemented: function '", #func, "' in class: '",      \
                  typeid(*this).name(), "'");                                 \
  }
  VIRTUAL_NOT_IMPLEMENTED(bool this_json_default())
  VIRTUAL_NOT_IMPLEMENTED(bool this_json_null())
  VIRTUAL_NOT_IMPLEMENTED(bool this_json_bool(bool /*b*/))
  VIRTUAL_NOT_IMPLEMENTED(bool this_json_int(int /*i*/))
  VIRTUAL_NOT_IMPLEMENTED(bool this_json_uint(unsigned /*i*/))
  VIRTUAL_NOT_IMPLEMENTED(bool this_json_int64(int64_t /*i*/))
  VIRTUAL_NOT_IMPLEMENTED(bool this_json_uint64(uint64_t /*i*/))
  VIRTUAL_NOT_IMPLEMENTED(bool this_json_double(double /*d*/))
  VIRTUAL_NOT_IMPLEMENTED(bool this_json_raw_number(const Ch * /*str*/,
                                                    SizeType /*length*/,
                                                    bool /*copy*/))
  VIRTUAL_NOT_IMPLEMENTED(bool this_json_string(const Ch * /*str*/,
                                                SizeType /*length*/,
                                                bool /*copy*/))
  VIRTUAL_NOT_IMPLEMENTED(bool this_json_start_object())
  VIRTUAL_NOT_IMPLEMENTED(bool this_json_key(const Ch * /*str*/,
                                             SizeType /*length*/,
                                             bool /*copy*/))
  VIRTUAL_NOT_IMPLEMENTED(bool this_json_end_object(SizeType /*memberCount*/))
  VIRTUAL_NOT_IMPLEMENTED(bool this_json_start_array())
  VIRTUAL_NOT_IMPLEMENTED(bool this_json_end_array(SizeType /*elementCount*/))
#undef VIRTUAL_NOT_IMPLEMENTED
};
template <class TResult, class TElementParser>
template <class... TArgs>
Abstract_Json_Array_Parser_With_Skip<TResult, TElementParser>::
  Abstract_Json_Array_Parser_With_Skip(
    bool skip, const std::function<void(TResult &&)> &on_parsed,
    const std::function<void()> &on_skipped,
    const std::function<bool(size_t index)> &skip_element_func,
    TArgs &&...element_parser_args)
    : base_type(skip, on_parsed, on_skipped),
      skip_element([this, skip_element_func](size_t index) {
        return this->skip || skip_element_func(index);
      }),
      element_parser(
        this->skip_element(0),
        [this](element_type &&value) {
          this->on_element_parsed(std::forward<element_type>(value),
                                  this->index);
          ++this->index;
          this->element_parser.reset(this->skip_element(this->index));
        },
        [this] {
          this->on_element_skipped(this->index);
          ++this->index;
          this->element_parser.reset(this->skip_element(this->index));
        },
        std::forward<TArgs>(element_parser_args)...)
{}
