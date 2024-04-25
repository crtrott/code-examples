# Contraints, Mandates and Constexpr

How modern C++ helps write safer and more maintainable software


### What is this about?

* Modern (C++20 and beyond) software design that helps catch mistakes earlier
* A primer on how the ISO C++ standard expresses design
* Learn by example using `std::mdspan`

***

|  
|  
|  

***


### mdspan - a comparison to Kokkos::View

`std::mdspan` is a C++ 23 facility based on `Kokkos::View`

* Difference: it is non-owning and doesn't allocate
  * `mdspan` is like an **unmanaged** `View`!

```c++
template<class ElementType, class Extents, class Layout, class Accessor>
class mdspan;

template<class DataType, class Layout, class MemorySpace, class MemoryTraits>
class View
```

***

|  
|  
|  

***


#### DataType

* `DataType` is a split into `ElementType` and `Extents`
* `double*[5]` becomes `double` and `extents<int, dynamic_extent, 5>`
* `mdspan` allows arbitrary mixing of compiler vs runtime extents
* `extents` has a mandatory `index_type` specifier
* `dextents<IndexType, Rank>` shortcut for all dynamic extents

***

|  
|  
|  

***


#### Layout

* layouts work the same
* `Kokkos::LayoutLeft` => `std::layout_left` or `std::layout_left_padded<dynamic_extent>`
* `Kokkos::LayoutRight` => `std::layout_right` or `std::layout_right_padded<dynamic_extent>`
* `Kokkos::LayoutStride` => `std::layout_stride` or `std::layout_left_padded<dynamic_extent>`
* this is a customization point: write your own layouts - now well defined how ...
* default layout for `mdspan` is `layout_right`

***

|  
|  
|  

***


#### MemorySpace and MemoryTraits

* these can be expressed via `Accessor`
* ISO C++23 comes with only `default_accessor` => `Kokkos::AnonymousSpace` + `Kokkos::Unmanaged`
* ISO C++26 will likely add `atomic_accessor`, `aligned_accessor` and some linear algebra things
* this is a customization point: write your own accessors - now well defined how ...
  * we will provide in Kokkos memory space aware accessors etc.

***

|  
|  
|  

***


#### Some examples

Using unmanaged views here ...

Check on GodBolt: [View vs mdspan example](https://godbolt.org/z/38nvTbGrn)

```c++
Kokkos::View<int**, Kokkos::LayoutLeft> a(ptr, N, M);
std::mdspan<int, std::dextents<int, 2>, std::layout_left> b(ptr, N,M);

assert((a(3,5) == b[3,5]));
assert((a.extent(0) == b.extent(0)));
static_assert(a.rank() == b.rank());
```

Both have `layout_right` as the default (for `View` only in Host build!!)
```c++
Kokkos::View<int[10][20]> a(ptr);
std::mdspan<int, std::extents<int, 10, 20>> b(ptr);

assert((a(3,5) == b[3,5]));
assert((a.extent(0) == b.extent(0)));
static_assert(a.static_extent(0) == b.static_extent(0));
static_assert(a.static_extent(1) == b.static_extent(1));
static_assert(a.rank() == b.rank());
```

Full specification: [ISO C++ Draft containers.views.multidim](https://eel.is/c++draft/views.multidim)

***

|  
|  
|  

***

### MDSpan Support in Kokkos

* We are shipping C++23 `mdspan` in the Kokkos namespace
* compatible with C++17 and C++20
  * biggest difference: use `()` operator instead of `[]` operator for data access
* Device enabled
* C++26 capabilities (`submdspan`, `padded` layouts, etc.) upcoming (likely Kokkos 4.4)
* Also Kokkos 4.4: Interoperability betwen `mdspan` and `View`:
  * Construct `mdspan` from `View`
  * Construct (unmanaged) `View` from `mdspan`
* Post Kokkos 4.4: support for `mdspan` in functions such as `deep_copy` etc.

***

|  
|  
|  

***


## ISO C++ Elements of Library Function Specification

ISO C++ uses 12 elements: *Constraints, Mandates, Preconditions, Effects, Synchronization, Postconditions, Result, Returns, Throws, Complexity, Remarks,* and *Error*.

Here I want to look at the ones which express limitations and requirements for users:

* **Constraints**: conditions for participating in overload resolution - i.e. do caller arguments match this overload
* **Mandates**: compile time conditions evaluated if *Constraints* are matched
* **Preconditions**: runtime conditions

For a description of all see: [ISO C++ Draft structure.specifications](https://eel.is/c++draft/structure#specifications-3)

***

|  
|  
|  

***


#### Start with an example

Lets look at a function which adds one `mdspan` to another for rank-1:

```c++
template<class T1, class E1, class L1, class A1,
         class T2, class E2, class L2, class A2>
void add(mdspan<T1,E1,L1,A1> dest, mdspan<T2,E2,L2,A2> src) {
  for(int i0; i0<dest.extent(0); i0++)
    dest[i0] += src[i0];
}
```

Calling it with a rank-2 `mdspan` would result in compile time error ([Godbolt Rank-2 Compile Time Error](https://godbolt.org/z/9zqr9hEzP)):

```
<source>:8:5: error: no viable overloaded operator[] for type 'std::mdspan<int, extents<unsigned long, 18446744073709551615, 18446744073709551615>, layout_right, default_accessor<int>>'
    8 |     dest[i0] += src[i0];
```

***

|  
|  
|  

***


### Using Mandates for a better error message

This is a place for a **Mandate**! In ISO C++ We would spell it like this:

```c++
template<class T1, class E1, class L1, class A1,
         class T2, class E2, class L2, class A2>
void add(mdspan<T1,E1,L1,A1> dest, mdspan<T2,E2,L2,A2> src);
```

* Mandates:
  * `dest.rank() == 1` is `true`, and
  * `src.rank() == 1` is `true`.
* Effects: For every integral `i` in the range [0, `dest.extent(0)`) performs `dest[i] += src[i]`. 

And implement it via `static_assert`:
```c++
template<class T1, class E1, class L1, class A1,
         class T2, class E2, class L2, class A2>
void add(mdspan<T1,E1,L1,A1> dest, mdspan<T2,E2,L2,A2> src) {
  static_assert(dest.rank() == 1, "add(dest,src): dest must have rank()==1");
  static_assert(src.rank() == 1, "add(dest,src): src must have rank()==1");
  for(int i0; i0<dest.extent(0); i0++)
    dest[i0] += src[i0];
}
```

Now the error message is:
```
<source>:7:17: error: static assertion failed due to requirement 'dest.rank() == 1': add(dest,src): dest must have rank()==1
    7 |   static_assert(dest.rank()==1, "add(dest,src): dest must have rank()==1");
      |                 ^~~~~~~~~~~~~~
<source>:38:7: note: in instantiation of function template specialization 'add<int, std::extents<unsigned long, 18446744073709551615, 18446744073709551615>, std::layout_right, std::default_accessor<int>, int, std::extents<unsigned long, 18446744073709551615, 18446744073709551615>, std::layout_right, std::default_accessor<int>>' requested here
   38 |       add(a,b);
      |       ^
<source>:7:28: note: expression evaluates to '2 == 1'
    7 |   static_assert(dest.rank()==1, "add(dest,src): dest must have rank()==1");
      |
```

***

|  
|  
|  

***


### Enabling add for vectors and matrices via constraints

To enable it for both rank-1 and rank-2 we need two overloads: in the ISO C++ specification we use **Constraints**:

```c++
template<class T1, class E1, class L1, class A1,
         class T2, class E2, class L2, class A2>
void add(mdspan<T1,E1,L1,A1> dest, mdspan<T2,E2,L2,A2> src);
```
* Constraints:
  * `dest.rank() == 1` is `true`, and
  * `src.rank() == 1` is `true`.
* Effects: For every integral `i` in the range [0, `dest.extent(0)`) performs `dest[i] += src[i]`. 

```c++
template<class T1, class E1, class L1, class A1,
         class T2, class E2, class L2, class A2>
void add(mdspan<T1,E1,L1,A1> dest, mdspan<T2,E2,L2,A2> src);
```
* Constraints:
  * `dest.rank() == 2` is `true`, and
  * `src.rank() == 2` is `true`.
* Effects: For every multi-dimensional index in `i` in `dest.extents()` perform `dest[i...] += src[i...]`. 

***

|  
|  
|  

***


In C++17 we can achieve constraints for a single function overlaod exploiting SFINAE (Substitution failure is not an error):

```c++
template<class T1, class E1, class L1, class A1,
         class T2, class E2, class L2, class A2,
         class Enable = std::enable_if_t<E1::rank()==1 && E2::rank()==1, void*>>
void add(std::mdspan<T1,E1,L1,A1> dest, std::mdspan<T2,E2,L2,A2> src)
```

Modern clang actually knows this pattern and already gives a decent-ish error message [SFINAE - Constraint Rank-1](https://godbolt.org/z/8acGn6T1q):
```c++
<source>:45:7: error: no matching function for call to 'add'
   45 |       add(a,b);
      |       ^~~
<source>:7:6: note: candidate template ignored: requirement 'extents<unsigned long, 18446744073709551615, 18446744073709551615>::rank() == 1' was not satisfied [with T1 = int, E1 = std::extents<unsigned long, 18446744073709551615, 18446744073709551615>, L1 = std::layout_right, A1 = std::default_accessor<int>, T2 = int, E2 = std::extents<unsigned long, 18446744073709551615, 18446744073709551615>, L2 = std::layout_right, A2 = std::default_accessor<int>]
    7 | void add(std::mdspan<T1,E1,L1,A1> dest, std::mdspan<T2,E2,L2,A2> src) {
      |      ^
1 error generated.
```

***

|  
|  
|  

***


But to get something which works for both we need to jump through hoops, the easiest way is to add a defaulted argument (not template argument):

```c++
template<class T1, class E1, class L1, class A1,
         class T2, class E2, class L2, class A2>
void add(std::mdspan<T1,E1,L1,A1> dest, std::mdspan<T2,E2,L2,A2> src,
         std::enable_if_t<(E1::rank()==1 && E2::rank()==1), void*> = nullptr) {
  for(int i0; i0<dest.extent(0); i0++)
    dest[i0] += src[i0];        
}

template<class T1, class E1, class L1, class A1,
         class T2, class E2, class L2, class A2>
void add(std::mdspan<T1,E1,L1,A1> dest, std::mdspan<T2,E2,L2,A2> src,
         std::enable_if_t<(E1::rank()==2 && E2::rank()==2), void*> = nullptr) {
  for(int i0; i0<dest.extent(0); i0++)
    for(int i1; i1<dest.extent(0); i1++)
      dest[i0, i1] += src[i0, i1];
}
```

[Godbolt](https://godbolt.org/z/e7Kz55zn3)

The nasty thing is that both require some extra arguments: either template or function argument.

*C++20 fixes this with the* `requires` *clause*!

***

|  
|  
|  

***



#### Constraints with requires

`requires` is part of C++ concepts and there are many things you can do with it.

Here I only want to look at some simple usage.

Instead of SFINAE we simply spell the requirement as a condition [GodBolt use requires clause](https://godbolt.org/z/KjWbPq6M9):


```c++
template<class T1, class E1, class L1, class A1,
         class T2, class E2, class L2, class A2>
requires(E1::rank()==1 && E2::rank()==1)
void add(std::mdspan<T1,E1,L1,A1> dest, std::mdspan<T2,E2,L2,A2> src) {
  for(int i0; i0<dest.extent(0); i0++)
    dest[i0] += src[i0];        
}

template<class T1, class E1, class L1, class A1,
         class T2, class E2, class L2, class A2>
requires(E1::rank()==2 && E2::rank()==2)
void add(std::mdspan<T1,E1,L1,A1> dest, std::mdspan<T2,E2,L2,A2> src) {
  for(int i0; i0<dest.extent(0); i0++)
    for(int i1; i1<dest.extent(0); i1++)
      dest[i0, i1] += src[i0, i1];
}
```

Now the error message without the second overload is even better saying that the constraint was not satisfied, and listing the specific constraint:

```
<source>:48:7: error: no matching function for call to 'add'
   48 |       add(a,b);
      |       ^~~
<source>:7:6: note: candidate template ignored: constraints not satisfied [with T1 = int, E1 = std::extents<unsigned long, 18446744073709551615, 18446744073709551615>, L1 = std::layout_right, A1 = std::default_accessor<int>, T2 = int, E2 = std::extents<unsigned long, 18446744073709551615, 18446744073709551615>, L2 = std::layout_right, A2 = std::default_accessor<int>]
    7 | void add(std::mdspan<T1,E1,L1,A1> dest, std::mdspan<T2,E2,L2,A2> src) {
      |      ^
<source>:6:10: note: because 'extents<unsigned long, 18446744073709551615, 18446744073709551615>::rank() == 1' (2 == 1) evaluated to false
    6 | requires(E1::rank()==1 && E2::rank()==1)
      |          ^
```

***

|  
|  
|  

***


Another slightly more elaborate overload set, with scalar add: [https://godbolt.org/z/188x4d33h](https://godbolt.org/z/188x4d33h).

It uses a more elaborate requires clause with pure template function parameters constraint via `requires`:

```c++
template<class T1, class T2>
requires(
    T1::rank()==1 && T2::rank()==1 &&
    is_mdspan_v<T1> && is_mdspan_v<T2>
)
void add(T1 dest, T2 src) {
  for(int i0; i0<dest.extent(0); i0++)
    dest[i0] += src[i0];        
}

template<class T1, class T2>
requires(!is_mdspan_v<T1>)
void add(T1& v1, T2 v2) {
  v1 += v2;
}
```

***

|  
|  
|  

***


#### Constraint order matters

In the above examples we actually made a slight mistake: constraints are evaluated in order!

If the scalar overload doesn't exist and `T1` is not an `mdspan` the error message is bad:

```
<source>:15:5: note: because substituted constraint expression is ill-formed: type 'int' cannot be used prior to '::' because it has no members
   15 |     T1::rank()==1 && T2::rank()==1 &&
```

Switching the order fixes that, i.e.

```
template<class T1, class T2>
requires(
    is_mdspan_v<T1> && is_mdspan_v<T2> &&
    T1::rank()==1 && T2::rank()==1
)
void add(T1 dest, T2 src) {
  for(int i0; i0<dest.extent(0); i0++)
    dest[i0] += src[i0];        
}
```

Resulting in:

```
<source>:15:5: note: because 'is_mdspan_v<int>' evaluated to false
   15 |     is_mdspan_v<T1> && is_mdspan_v<T2> &&
```

See here for both errors: [Godbolt errors depend on constraint order](https://godbolt.org/z/oT6W4dsz3)

***

|  
|  
|  

***


#### Checking for well formed expressions

We can also check that the `+=` actually works (e.g. you don't call the function with `mdspan` of `std::array`):

```c++
template<class T1, class T2>
void add(T1 dest, T2 src)
requires(
    is_mdspan_v<T1> && is_mdspan_v<T2> &&
    T1::rank()==1 && T2::rank()==1 &&
    requires{dest[0] += dest[0];}
) {
  for(int i0; i0<dest.extent(0); i0++)
    dest[i0] += src[i0];        
}
```

Note: the switch of order (requires after the function signature) to be able to use `dest` and `src`.
Here is the [Godbolt code with check for addable](https://godbolt.org/z/aMq7EMe3e)

***

|  
|  
|  

***


### Adding preconditions

*Preconditions* are for runtime checks. It is the responsibility of the user to not violate them - otherwise behavior is undefined.

This is in contrast to *Mandates* where it is the implementations responsibility to catch them at compile time!

An implementation should check *Preconditions* at a minimum in debug mode!

So lets check the dimensions match:

```c++
template<class T1, class T2>
void add(T1 dest, T2 src)
requires(
    is_mdspan_v<T1> && is_mdspan_v<T2> &&
    T1::rank()==1 && T2::rank()==1 &&
    requires{dest[0] += dest[0];}
) {
  assert(dest.extent(0) == src.extent(0));
  for(int i0; i0<dest.extent(0); i0++)
    dest[i0] += src[i0];
}
```

But sometimes we can check the `extents` at compile time! We should catch that if possible!

***

|  
|  
|  

***


It doesn't make much sense to do it as a *Constraint* - nobody wants a different overload of `add` for that.

```c++
template<class T1, class T2>
void add(T1 dest, T2 src)
requires(
    is_mdspan_v<T1> && is_mdspan_v<T2> &&
    T1::rank()==1 && T2::rank()==1 &&
    requires{dest[0] += dest[0];}
) {
  static_assert(
    (T1::static_extent(0) == std::dynamic_extent) ||
    (T2::static_extent(0) == std::dynamic_extent) ||
    (T1::static_extent(0) == T2::static_extent(1)),
    "Mismatching static extents are not allowed"
  );
  assert(dest.extent(0) == src.extent(0));
  for(int i0; i0<dest.extent(0); i0++)
    dest[i0] += src[i0];
}
```

See the error here: [Godbolt with mismatching static extents](https://godbolt.org/z/bG7fj3GW9)


***

|  
|  
|  

***

### The complete specification in ISO C++ style:

```c++
template<class T1, class T2>
void add(T1 dest, T2 src);
```

* *Constraints:* 
  * `is_mdspan_v<T1>` is `true`.
  * `is_mdspan_v<T2>` is `true`.
  * `T1::rank()==1` is `true`.
  * `T2::rank()==1` is `true`.
  * `dest[0] += src[0]` is well formed.
* *Mandates:* If neither `T1::static_extent(0)` nor `T2::static_extent(0)` is `dynamic_extent`,
   then `T1::static_extent(0) == T2::static_extent(0)` is `true`.
* *Preconditions:* `dest.extent(0) == src.extent(0)` is `true`.
* *Effects:* For every integral `i` in the range [0, `dest.extent(0)`) performs `dest[i] += src[i]`. 

And here is the code: [Godbolt with all the things](https://godbolt.org/z/5n38ovo9z).

***

|  
|  
|  

***


### Constraints and Mandates for Constructors

All the above also applies to constructors - however there is one big issue: *Mandates* are not taking into account by traits such as `is_constructible_v`!

Consider the following matrix class:


```c++
template<class T, size_t N, size_t M>
struct Matrix {
  T data[N*M];
  std::mdspan<T, std::extents<int, N, M>> v;
  template<class U, class E, class L, class A>
  Matrix(std::mdspan<U, E, L, A> d):v(data) {
    for(int i=0; i<N; i++)
      for(int j=0; j<N; j++)
        v[i,j] = d[i,j];

  }
}; 
```

You may want to have mandates as the ones we had for `add` in the constructor such as:

```c++
  static_assert(
    (E::static_extent(0) == std::dynamic_extent) ||
    (E::static_extent(0) == N),
    "Mismatching static extents are not allowed"
  );
```

However, `is_constructible_v<Matrix<float, 5,5>, mdspan<float, extents<int, 4,4>>>` is `true`, even though

```c++
std::mdspan<float, std::extents<int, 4,4>> v(ptr1);
Matrix<float, 5,5> m(v);
```

will not compile [Godbolt Constructor with static assert](https://godbolt.org/z/esfcd3dxe)!


**Mandates in constructors (i.e. static_assert) are strongly discouraged!**

***

|  
|  
|  

***


#### Preconditions for constructors and explicit

Another thing to check in the above constructor is a precondition whether a runtime size matches the `Matrix` size:

```c++
assert(v.extent(0) == N);
```

*Best Practice:* Make constructors that could result in a throw (conditionally) explicit!

```c++
template<class U, class E, class L, class A>
explicit(E::static_extent(0) == std::dynamic_extent ||
         E::static_extent(1) == std::dynamic_extent)
Matrix(std::mdspan<U, E, L, A> d)i
```

And here is the full thing with constraints and precondition [Godbolt Matrix Class Example](https://godbolt.org/z/rhhvx14fM)


***

|  
|  
|  

***


## Summary Constraints, Mandates, Preconditions

* *Constraints*: build overload sets - compile time check, is this call valid?
  * C++20 use `requires` clause
  * C++17 implemented via SFINAE
* *Mandates*: catch errors inside functions one never would want to cover with different overload
  * `static_assert` inside the function
  * Careful in constructors - generally not recommended because `is_constructible_v` does not take this into account
* *Preconditions*: catch runtime errors
  * `assert` or `throw` inside the function
  * for constructors: often good to have matchin conditional `explicit` clause! 


***

|  
|  
|  

***

## if constexpr as an alternative to overload sets

There is an alternative to overload sets: `if constexpr`

This enables you to have finegrained code sections which get compiled conditionally within a single block.

Lets look at the `add` function with that approach:

```c++
template<class T1, class T2>
requires(
    is_mdspan_v<T1> && is_mdspan_v<T2> &&
    T1::rank()==T2::rank()
)
void add(T1 dest, T2 src) {
  static_assert(T1::rank() == 1 || T1::rank() == 2);

  if constexpr (T1::rank() == 1) {
    assert(dest.extent(0) == src.extent(0));
    for(int i0; i0<dest.extent(0); i0++)
      dest[i0] += src[i0];
  } else {
    assert(dest.extent(0) == src.extent(0));
    assert(dest.extent(1) == src.extent(1));
    for(int i0; i0<dest.extent(0); i0++)
      for(int i1; i1<dest.extent(1); i1++)
        dest[i0, i1] += src[i0, i1];
  }
}
```

* Condition inside `if constexpr` must be constant evaluable
* `if` - `else if` - `else` nesting is fine

Code: [Godbolt if constexpr](https://godbolt.org/z/fsvEEqWb1)

***

|  
|  
|  

***


## Some thoughts on constexpr as function attribute

Consider a simplified version of `std::extents`:

```c++
template<size_t N>
struct exts {
  constexpr static size_t static_extent() { return N; }
  constexpr size_t extent() const { return N; }
};

template<>
struct exts<std::dynamic_extent> {
  size_t N;
  constexpr exts(size_t val):N(val) {};
  constexpr static size_t static_extent() { return std::dynamic_extent; }
  constexpr size_t extent() const { return N; }
};
```

Everything in here is `constexpr` - that doesn't mean you can always use every function in constant expressions!
  
  
`constexpr` means: you can use it in constant expressions if all the inputs are available at compile time!

Full code exmaple for this and following: [Godbolt constexpr explanation](https://godbolt.org/z/eb3jv7aK7)


***

|  
|  
|  

***

### Fully static use case

When you use the fully static version, both `static_extent` and `extent` can be used as constant expressions:

```c++
    exts<5> ext_static;
    if constexpr (ext_static.static_extent()==5) {
        printf("static 1\n");
    }
    if constexpr (ext_static.extent()==5) {
        printf("static 2\n");
    }
```

* You also can use both functions as template arguments


***

|  
|  
|  

***

### Dynamic use case

If something is dynamic you can't do that - depending on the place not even if the constructor argument was compile time known:

```c++
    exts<std::dynamic_extent> ext_dynamic(5);
    if constexpr (ext_dynamic.static_extent()==5) {
        printf("static 1\n");
    }
    // this doesn't work
    if constexpr (ext_dynamic.static_extent()==5) {
        printf("static 1\n");
    }
```

But you can if its inside a wrapped in another contexpr function:

```c++
template<class Ext>
constexpr size_t foo() {
  Ext e(5);
  return e.extent();
};
```

Now this works:
```c++
    if constexpr (foo<decltype(ext_dynamic)>()==5) {
        printf("dynamic not really\n");
    }
```


