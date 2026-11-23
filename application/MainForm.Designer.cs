namespace BiofoulingSegmentation;

partial class MainForm
{
    private System.ComponentModel.IContainer components = null;

    protected override void Dispose(bool disposing)
    {
        if (disposing && (components != null))
        {
            components.Dispose();
        }
        base.Dispose(disposing);
    }

    #region Windows Form Designer generated code

    private void InitializeComponent()
    {
        inputFolderTextBox = new TextBox();
        selectInputFolderButton = new Button();
        outputFolderTextBox = new TextBox();
        selectOutputFolderButton = new Button();
        modelPathTextBox = new TextBox();
        selectModelButton = new Button();
        startButton = new Button();
        progressBar = new ProgressBar();
        statusStrip = new StatusStrip();
        statusLabel = new ToolStripStatusLabel();
        statusStrip.SuspendLayout();
        SuspendLayout();
        //
        // inputFolderTextBox
        //
        inputFolderTextBox.Location = new Point(125, 30);
        inputFolderTextBox.Name = "inputFolderTextBox";
        inputFolderTextBox.Size = new Size(350, 27);
        inputFolderTextBox.TabIndex = 0;
        //
        // selectInputFolderButton
        //
        selectInputFolderButton.Location = new Point(200, 75);
        selectInputFolderButton.Name = "selectInputFolderButton";
        selectInputFolderButton.Size = new Size(200, 40);
        selectInputFolderButton.TabIndex = 1;
        selectInputFolderButton.Text = "Select Input Folder";
        selectInputFolderButton.UseVisualStyleBackColor = true;
        selectInputFolderButton.Click += SelectInputFolderButton_Click;
        //
        // outputFolderTextBox
        //
        outputFolderTextBox.Location = new Point(125, 150);
        outputFolderTextBox.Name = "outputFolderTextBox";
        outputFolderTextBox.Size = new Size(350, 27);
        outputFolderTextBox.TabIndex = 2;
        //
        // selectOutputFolderButton
        //
        selectOutputFolderButton.Location = new Point(200, 195);
        selectOutputFolderButton.Name = "selectOutputFolderButton";
        selectOutputFolderButton.Size = new Size(200, 40);
        selectOutputFolderButton.TabIndex = 3;
        selectOutputFolderButton.Text = "Select Output Folder";
        selectOutputFolderButton.UseVisualStyleBackColor = true;
        selectOutputFolderButton.Click += SelectOutputFolderButton_Click;
        //
        // modelPathTextBox
        //
        modelPathTextBox.Location = new Point(125, 270);
        modelPathTextBox.Name = "modelPathTextBox";
        modelPathTextBox.Size = new Size(350, 27);
        modelPathTextBox.TabIndex = 4;
        //
        // selectModelButton
        //
        selectModelButton.Location = new Point(200, 315);
        selectModelButton.Name = "selectModelButton";
        selectModelButton.Size = new Size(200, 40);
        selectModelButton.TabIndex = 5;
        selectModelButton.Text = "Provide Model Path";
        selectModelButton.UseVisualStyleBackColor = true;
        selectModelButton.Click += SelectModelButton_Click;
        //
        // startButton
        //
        startButton.BackColor = Color.FromArgb(192, 192, 255);
        startButton.FlatAppearance.BorderColor = Color.Black;
        startButton.FlatAppearance.BorderSize = 2;
        startButton.FlatStyle = FlatStyle.Flat;
        startButton.Font = new Font("Segoe UI", 9F, FontStyle.Bold);
        startButton.Location = new Point(200, 430);
        startButton.Name = "startButton";
        startButton.Size = new Size(200, 50);
        startButton.TabIndex = 6;
        startButton.Text = "Initiate Predictions";
        startButton.UseVisualStyleBackColor = false;
        startButton.Click += StartButton_Click;
        //
        // progressBar
        //
        progressBar.Dock = DockStyle.Bottom;
        progressBar.Location = new Point(0, 503);
        progressBar.Name = "progressBar";
        progressBar.Size = new Size(582, 24);
        progressBar.Style = ProgressBarStyle.Continuous;
        progressBar.TabIndex = 7;
        //
        // statusStrip
        //
        statusStrip.ImageScalingSize = new Size(20, 20);
        statusStrip.Items.AddRange(new ToolStripItem[] { statusLabel });
        statusStrip.Location = new Point(0, 527);
        statusStrip.Name = "statusStrip";
        statusStrip.Size = new Size(582, 26);
        statusStrip.TabIndex = 8;
        //
        // statusLabel
        //
        statusLabel.Name = "statusLabel";
        statusLabel.Size = new Size(50, 20);
        statusLabel.Text = "Ready";
        //
        // MainForm
        //
        AutoScaleDimensions = new SizeF(8F, 20F);
        AutoScaleMode = AutoScaleMode.Font;
        BackColor = Color.LightGray;
        ClientSize = new Size(582, 553);
        Controls.Add(progressBar);
        Controls.Add(statusStrip);
        Controls.Add(startButton);
        Controls.Add(selectModelButton);
        Controls.Add(modelPathTextBox);
        Controls.Add(selectOutputFolderButton);
        Controls.Add(outputFolderTextBox);
        Controls.Add(selectInputFolderButton);
        Controls.Add(inputFolderTextBox);
        Font = new Font("Segoe UI", 10F);
        FormBorderStyle = FormBorderStyle.FixedSingle;
        MaximizeBox = false;
        Name = "MainForm";
        Text = "Marine Biofouling Segmentation";
        statusStrip.ResumeLayout(false);
        statusStrip.PerformLayout();
        ResumeLayout(false);
        PerformLayout();
    }

    #endregion

    private TextBox inputFolderTextBox;
    private Button selectInputFolderButton;
    private TextBox outputFolderTextBox;
    private Button selectOutputFolderButton;
    private TextBox modelPathTextBox;
    private Button selectModelButton;
    private Button startButton;
    private ProgressBar progressBar;
    private StatusStrip statusStrip;
    private ToolStripStatusLabel statusLabel;
}
