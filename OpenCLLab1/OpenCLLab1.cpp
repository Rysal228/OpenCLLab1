#include <stdio.h>
#include <string.h>
#include <windows.h>
#include <iostream>
#include <locale.h>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
int main()
{
    setlocale(LC_ALL, "Russian");

    cl_int err;

    // ============================================================
    // 1. Получение OpenCL-платформ
    // ============================================================

    cl_uint platformCount = 0;

    err = clGetPlatformIDs(0, nullptr, &platformCount);

    if (err != CL_SUCCESS || platformCount == 0)
    {
        printf("Ошибка: OpenCL-платформы не найдены.\n");
        return -1;
    }

    printf("Определено OpenCL платформ: %u\n", platformCount);

    cl_platform_id* platforms = new cl_platform_id[platformCount];

    err = clGetPlatformIDs(
        platformCount,
        platforms,
        nullptr
    );

    if (err != CL_SUCCESS)
    {
        printf("Ошибка получения списка платформ: %d\n", err);

        delete[] platforms;
        return -1;
    }


    // ============================================================
    // 2. Выбор NVIDIA-платформы
    // ============================================================

    cl_platform_id selectedPlatform = nullptr;

    for (cl_uint i = 0; i < platformCount; i++)
    {
        char vendor[256] = {};

        clGetPlatformInfo(
            platforms[i],
            CL_PLATFORM_VENDOR,
            sizeof(vendor),
            vendor,
            nullptr
        );

        printf("Платформа %u: %s\n", i + 1, vendor);

        if (selectedPlatform == nullptr &&
            strstr(vendor, "NVIDIA") != nullptr)
        {
            selectedPlatform = platforms[i];
        }
    }

    if (selectedPlatform == nullptr)
    {
        printf("Ошибка: NVIDIA OpenCL-платформа не найдена.\n");

        delete[] platforms;
        return -1;
    }

    printf("\nВыбрана NVIDIA OpenCL-платформа.\n");


    // ============================================================
    // 3. Получение устройства
    // ============================================================

    cl_uint deviceCount = 0;

    err = clGetDeviceIDs(
        selectedPlatform,
        CL_DEVICE_TYPE_GPU,
        0,
        nullptr,
        &deviceCount
    );

    if (err != CL_SUCCESS || deviceCount == 0)
    {
        printf("Ошибка: NVIDIA GPU OpenCL-устройство не найдено.\n");

        delete[] platforms;
        return -1;
    }

    cl_device_id* devices = new cl_device_id[deviceCount];

    err = clGetDeviceIDs(
        selectedPlatform,
        CL_DEVICE_TYPE_GPU,
        deviceCount,
        devices,
        nullptr
    );

    if (err != CL_SUCCESS)
    {
        printf("Ошибка получения OpenCL-устройства: %d\n", err);

        delete[] devices;
        delete[] platforms;
        return -1;
    }

    char deviceName[256] = {};

    clGetDeviceInfo(
        devices[0],
        CL_DEVICE_NAME,
        sizeof(deviceName),
        deviceName,
        nullptr
    );

    printf("Выбрано устройство: %s\n", deviceName);


    // ============================================================
    // 4. Создание контекста
    // ============================================================

    cl_context context = clCreateContext(
        nullptr,
        1,
        &devices[0],
        nullptr,
        nullptr,
        &err
    );

    if (err != CL_SUCCESS || context == nullptr)
    {
        printf("Ошибка создания контекста: %d\n", err);

        delete[] devices;
        delete[] platforms;
        return -1;
    }

    printf("Контекст успешно создан.\n");


    // ============================================================
    // 5. Создание очереди команд
    // ============================================================

    cl_command_queue commandQueue = clCreateCommandQueue(
        context,
        devices[0],
        0,
        &err
    );

    if (err != CL_SUCCESS || commandQueue == nullptr)
    {
        printf("Ошибка создания очереди команд: %d\n", err);

        clReleaseContext(context);

        delete[] devices;
        delete[] platforms;

        return -1;
    }

    const char* opencl_source =
        "__kernel void Simple (__global int *n) "
        "{ *n = (*n) * 2; }\n";

    cl_program opencl_program = clCreateProgramWithSource(
        context,
        1,
        &opencl_source,
        nullptr,
        &err
    );

    err = clBuildProgram(
        opencl_program,
        1,
        devices,
        nullptr,
        nullptr,
        nullptr
    );

    cl_kernel kernel = clCreateKernel(
        opencl_program,
        "Simple",
        &err
    );

    int value = 10;

    cl_mem valueForGPU = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(int),
        nullptr,
        &err
    );

    clEnqueueWriteBuffer(
        commandQueue,
        valueForGPU,
        CL_TRUE,
        0,
        sizeof(int),
        &value,
        0,
        nullptr,
        nullptr
    );

    clSetKernelArg(
        kernel,
        0,
        sizeof(cl_mem),
        &valueForGPU
    );

    size_t globalWorkSize = 1;

    clEnqueueNDRangeKernel(
        commandQueue,
        kernel,
        1,
        nullptr,
        &globalWorkSize,
        nullptr,
        0,
        nullptr,
        nullptr
    );

    clFinish(commandQueue);

    clEnqueueReadBuffer(
        commandQueue,
        valueForGPU,
        CL_TRUE,
        0,
        sizeof(int),
        &value,
        0,
        nullptr,
        nullptr
    );

    printf("Результат: %d\n", value);

    // ============================================================
    // Завершение работы
    // ============================================================

    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);

    delete[] devices;
    delete[] platforms;

    return 0;
}