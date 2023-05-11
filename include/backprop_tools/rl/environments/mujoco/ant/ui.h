#ifndef BACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT_UI_H
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT_UI_H
#include <GLFW/glfw3.h>
namespace backprop_tools::rl::environments::mujoco::ant {
    template <typename T_ENVIRONMENT>
    struct UI{
        using ENVIRONMENT = T_ENVIRONMENT;
        ENVIRONMENT* env;
        GLFWwindow* window;
        mjvCamera camera;
        mjvOption option;
        mjvScene scene;
        mjrContext context;
        bool button_left = false;
        bool button_middle = false;
        bool button_right =  false;
        double lastx = 0;
        double lasty = 0;
    };
    namespace ui::callbacks{
        template <typename UI>
        void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
            UI* ui = (UI*)glfwGetWindowUserPointer(window);
            if (act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE) {
                mj_resetData(ui->env->model, ui->env->data);
                mj_forward(ui->env->model, ui->env->data);
            }
        }
        template <typename UI>
        void mouse_button(GLFWwindow* window, int button, int act, int mods) {
            UI* ui = (UI*)glfwGetWindowUserPointer(window);
            ui->button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
            ui->button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
            ui->button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

            glfwGetCursorPos(window, &ui->lastx, &ui->lasty);
        }

        template <typename UI>
        void mouse_move(GLFWwindow* window, double xpos, double ypos) {
            UI* ui = (UI*)glfwGetWindowUserPointer(window);
            if (!ui->button_left && !ui->button_middle && !ui->button_right) {
                return;
            }
            double dx = xpos - ui->lastx;
            double 