```
conan profile detect --force
```
In the repository directory:
```
conan install . --output-folder=build_conan --build=missing --settings=build_type=Debug
conan install . --output-folder=build_conan --build=missing --settings=build_type=Release
```

```
cd build_conan
```

`